# infer_medsam2_ultrasound.py
import argparse, os, time, nibabel as nib, numpy as np
import cv2, torch
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import ndimage
from ultrasound_data_loader import UltrasoundDataLoader

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_dir',  required=True,
                    help='Dir containing ultrasound data files (*PPM.nnrd)')
parser.add_argument('-o','--output_dir', required=True,
                    help='Dir to save *.nii.gz segmentations')
parser.add_argument('--ckpt_path',       default='checkpoints/MedSAM2_latest.pt')
parser.add_argument('--config_path',     default='sam2/configs',
                    help='Path to config directory')
parser.add_argument('--yaml',           default='sam2.1_hiera_t512',
                    help='Config name without .yaml extension')
parser.add_argument('--device',         default='cuda:0')
parser.add_argument('--prompt_mode',    default='mask+point', 
                    choices=['auto', 'center', 'box', 'mask', 'mask+point', 'mask+box', 'point_only', 'box_only', 'none'],
                    help='Prompt type: auto/center/box/mask/mask+point/mask+box/point_only/box_only/none')
parser.add_argument('--box_size',       default=100, type=int,
                    help='Size of center box for box prompts')
parser.add_argument('--mask_radius',    default=50, type=int,
                    help='Radius of center mask for mask prompts')
parser.add_argument('--gt_dir',         default=None,
                    help='Directory containing ground truth labels (*.nii.gz)')
parser.add_argument('--gt_suffix',      default='_gt.nii.gz',
                    help='Suffix for ground truth files (e.g., _gt.nii.gz)')
parser.add_argument('--prompt_from_gt', action='store_true',
                    help='Generate prompts from ground truth labels')
parser.add_argument('--data_structure', default='unet',
                    choices=['unet', 'us3d'],
                    help='Data structure: unet (volumes/labels) or us3d (volumes/labels/prompts)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Create data directories for US3D structure
if args.data_structure == 'us3d':
    os.makedirs(os.path.join(args.output_dir, 'volumes'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'prompts'), exist_ok=True)

# ---------- Build model ----------
print("Loading MedSAM2 model...")
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Use compose with config_name
# Change to config directory temporarily
import os
original_cwd = os.getcwd()
os.chdir(args.config_path)

try:
    cfg = compose(
        config_name=args.yaml,
    )
finally:
    # Restore original working directory
    os.chdir(original_cwd)
OmegaConf.resolve(cfg)
sam2_model = instantiate(cfg.model, _recursive_=True)

# Load checkpoint
if args.ckpt_path:
    import torch
    sd = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = sam2_model.load_state_dict(sd)
    print(f"Loaded checkpoint: {args.ckpt_path}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

sam2_model = sam2_model.to(args.device)
sam2_model.eval()
predictor = SAM2ImagePredictor(sam2_model)
print("Model loaded successfully!")

# ---------- Helper ----------
def preprocess_slice(img2d, side=512):
    """float32 2D  →  uint8 RGB 512×512"""
    img2d = (img2d - img2d.min()) / (img2d.ptp() + 1e-6) * 255
    img2d = img2d.astype(np.uint8)
    img2d = cv2.resize(img2d, (side, side),
                       interpolation=cv2.INTER_LINEAR)
    # Convert to RGB format (HWC)
    rgb = np.stack([img2d, img2d, img2d], axis=2)  # (H,W,3)
    return rgb

def calculate_center_mass_and_variance(label_512):
    """Calculate center of mass and variance for point prompt filtering"""
    if label_512.max() == 0:
        return None, None
    
    # Find connected components
    label_binary = (label_512 > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label_binary, connectivity=8)
    
    if num_labels <= 1:
        return None, None
    
    # Find largest component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]
    
    if largest_area < 10:
        return None, None
    
    # Get centroid
    centroid_x = int(centroids[largest_label][0])
    centroid_y = int(centroids[largest_label][1])
    
    # Calculate variance (spread) of the object
    y_coords, x_coords = np.where(labels == largest_label)
    if len(x_coords) > 0:
        variance = np.var(x_coords) + np.var(y_coords)
    else:
        variance = 0
    
    return (centroid_x, centroid_y), variance

def generate_box_from_label(label_512, variance_threshold=1000):
    """Generate bounding box from label with variance filtering"""
    if label_512.max() == 0:
        return None
    
    # Find connected components
    label_binary = (label_512 > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label_binary, connectivity=8)
    
    if num_labels <= 1:
        return None
    
    # Find largest component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]
    
    if largest_area < 10:
        return None
    
    # Get bounding box
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]
    
    # Calculate variance for filtering
    y_coords, x_coords = np.where(labels == largest_label)
    if len(x_coords) > 0:
        variance = np.var(x_coords) + np.var(y_coords)
        if variance < variance_threshold:  # Filter out very compact objects
            return None
    
    return np.array([x, y, x + w, y + h])

def generate_prompt_from_gt(gt_slice, prompt_mode='mask+point'):
    """Generate prompts from ground truth slice (UNet format compatible)"""
    H, W = gt_slice.shape
    
    # Resize label to 512x512 (MedSAM2 input size)
    gt_512 = cv2.resize(gt_slice, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Calculate center of mass and variance
    center_mass, variance = calculate_center_mass_and_variance(gt_512)
    
    if center_mass is None:
        return None
    
    centroid_x, centroid_y = center_mass
    
    if prompt_mode in ['auto', 'center', 'point_only']:
        # Point prompt at center of mass
        return {
            'point_coords': np.array([[centroid_x, centroid_y]]),
            'point_labels': np.array([1])
        }
    elif prompt_mode == 'box_only':
        # Box prompt only
        box = generate_box_from_label(gt_512)
        if box is None:
            return None
        return {'box': box}
    elif prompt_mode == 'box':
        # Box prompt around the object
        box = generate_box_from_label(gt_512)
        if box is None:
            return None
        return {'box': box}
    elif prompt_mode == 'mask':
        # Use the ground truth mask directly
        return {
            'mask_input': gt_512.astype(np.float32)
        }
    elif prompt_mode == 'mask+point':
        # Combined mask and point prompts (recommended)
        return {
            'mask_input': gt_512.astype(np.float32),
            'point_coords': np.array([[centroid_x, centroid_y]]),
            'point_labels': np.array([1])
        }
    elif prompt_mode == 'mask+box':
        # Combined mask and box prompts
        box = generate_box_from_label(gt_512)
        if box is None:
            return None
        return {
            'mask_input': gt_512.astype(np.float32),
            'box': box
        }
    
    return None

def get_gt_filename(input_filename, gt_dir, gt_suffix):
    """Get corresponding ground truth filename"""
    # Remove .nii.gz extension
    base_name = input_filename.replace('.nii.gz', '').replace('.nii', '')
    
    # Try different patterns
    possible_names = [
        f"{base_name}{gt_suffix}",
        f"{base_name}_label{gt_suffix}",
        f"{base_name}_mask{gt_suffix}",
        f"{base_name}_seg{gt_suffix}",
        f"{gt_suffix.replace('.nii.gz', '')}_{base_name}.nii.gz",
        f"gt_{base_name}.nii.gz"
    ]
    
    for name in possible_names:
        gt_path = os.path.join(gt_dir, name)
        if os.path.exists(gt_path):
            return gt_path
    
    return None

# ---------- Initialize data loader ----------
print("🔍 Initializing ultrasound data loader...")
data_loader = UltrasoundDataLoader(args.input_dir)

if not data_loader.data_files:
    print("❌ No data files found! Please check your data directory.")
    exit(1)

print(f"✅ Found {len(data_loader.data_files)} data-label pairs")

# ---------- Iterate volumes ----------
for data_file, label_file in tqdm(data_loader.data_files, desc='Processing volumes'):
    # Load volume and label
    vol_data, lbl_data = data_loader.load_volume_and_label(data_file, label_file)
    
    if vol_data is None or lbl_data is None:
        print(f"❌ Failed to load {data_file}")
        continue
    
    # Validate data
    if not data_loader.validate_data(vol_data, lbl_data):
        print(f"⚠️  Skipping invalid data: {data_file}")
        continue
    
    H, W, D = vol_data.shape
    seg_stack = np.zeros((H, W, D), np.uint8)
    
    # Get file info for naming
    file_info = data_loader.get_file_info(data_file)
    output_name = f"{file_info['base_name']}.nii.gz"

    for z in range(D):
        slc = vol_data[:,:,z]
        rgb_slice = preprocess_slice(slc)  # (H,W,3) uint8
        
        # Set image for predictor
        predictor.set_image(rgb_slice)
        
        # Generate prompt from label data (always available)
        gt_slice = lbl_data[:,:,z].astype(np.float32)
        gt_prompt = generate_prompt_from_gt(gt_slice, args.prompt_mode)
        
        # Generate prompts based on mode
        if gt_prompt is not None:
            # Use ground truth-based prompt
            masks, scores, logits = predictor.predict(
                **gt_prompt,
                multimask_output=False
            )
        elif args.prompt_mode == 'auto':
            # Use center point as prompt
            point_coords = np.array([[W//2, H//2]])  # Center point
            point_labels = np.array([1])  # Foreground point
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
        elif args.prompt_mode == 'center':
            # Manual center point (you can modify this)
            point_coords = np.array([[W//2, H//2]])
            point_labels = np.array([1])
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
        elif args.prompt_mode == 'box':
            # Use center box as prompt
            box_size = args.box_size
            x1 = max(0, W//2 - box_size//2)
            y1 = max(0, H//2 - box_size//2)
            x2 = min(W, W//2 + box_size//2)
            y2 = min(H, H//2 + box_size//2)
            box = np.array([x1, y1, x2, y2])
            masks, scores, logits = predictor.predict(
                box=box,
                multimask_output=False
            )
        elif args.prompt_mode == 'mask':
            # Use center circular mask as prompt
            mask_radius = args.mask_radius
            center_x, center_y = W//2, H//2
            
            # Create circular mask
            y, x = np.ogrid[:H, :W]
            mask = ((x - center_x)**2 + (y - center_y)**2 <= mask_radius**2).astype(np.float32)
            
            masks, scores, logits = predictor.predict(
                mask_input=mask,
                multimask_output=False
            )
        else:  # 'none'
            # No prompts - this might not work well with MedSAM2
            masks, scores, logits = predictor.predict(
                multimask_output=False
            )
        
        # Get the best mask
        if len(masks) > 0:
            best_mask = masks[np.argmax(scores)]
            out_np = best_mask.astype(np.uint8)
        else:
            out_np = np.zeros((512, 512), dtype=np.uint8)
        
        # Resize back to original dimensions
        out_np = cv2.resize(out_np, (W, H),
                            interpolation=cv2.INTER_NEAREST)
        seg_stack[:,:,z] = out_np

    # ---------- 保存 ----------
    if args.data_structure == 'us3d':
        # Save in US3D structure
        base_name = file_info['base_name']
        
        # Save volume
        vol_output_path = os.path.join(args.output_dir, 'volumes', output_name)
        vol_nib = nib.Nifti1Image(vol_data, affine=np.eye(4))
        nib.save(vol_nib, vol_output_path)
        
        # Save segmentation
        seg_output_path = os.path.join(args.output_dir, 'labels', output_name)
        seg_nib = nib.Nifti1Image(seg_stack.astype(np.uint8), affine=np.eye(4))
        nib.save(seg_nib, seg_output_path)
        
        # Save prompts (label data)
        prompt_output_path = os.path.join(args.output_dir, 'prompts', output_name)
        lbl_nib = nib.Nifti1Image(lbl_data, affine=np.eye(4))
        nib.save(lbl_nib, prompt_output_path)
    else:
        # Save in UNet structure
        seg_nib = nib.Nifti1Image(seg_stack.astype(np.uint8), affine=np.eye(4))
        nib.save(seg_nib, os.path.join(args.output_dir, output_name))
    
    print(f'[OK] {output_name}  →  {seg_stack.mean():.3f} foreground ratio')

print('All done!') 