#!/usr/bin/env python3
"""
Setup script for MedSAM2 fine-tuning on ultrasound data
Follows the UNet format workflow with intelligent prompts
"""

import os
import yaml
import shutil
from pathlib import Path

def create_finetune_config():
    """Create fine-tuning configuration for ultrasound data"""
    
    # Base config path
    base_config = "sam2/configs/sam2.1_hiera_tiny_finetune512.yaml"
    
    if not os.path.exists(base_config):
        print(f"❌ Base config not found: {base_config}")
        return False
    
    # Read base config
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update for ultrasound fine-tuning
    config['dataset'] = {
        'train_root': '/path/to/train_volumes',      # Update this
        'train_label_root': '/path/to/train_labels', # Update this
        'val_root': '/path/to/val_volumes',          # Update this
        'val_label_root': '/path/to/val_labels',     # Update this
        'modality': 'ultrasound',
        'prompt_type': 'mask+point',                 # Our strategy
        'image_size': 512,
        'batch_size': 1,                             # Single slice per batch
        'num_workers': 4
    }
    
    config['trainer'] = {
        'ckpt_path': 'checkpoints/MedSAM2_latest.pt',
        'epochs': 80,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,
        'save_freq': 10,
        'eval_freq': 5
    }
    
    config['model'] = {
        'sam_mask_decoder_extra_args': {
            'dynamic_multimask_via_stability': True,
            'dynamic_multimask_stability_delta': 0.05,
            'dynamic_multimask_stability_thresh': 0.98
        }
    }
    
    # Save updated config
    output_config = "sam2/configs/ultrasound_finetune.yaml"
    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created fine-tuning config: {output_config}")
    return output_config

def create_training_script():
    """Create training script for ultrasound fine-tuning"""
    
    script_content = '''#!/usr/bin/env python3
"""
MedSAM2 Ultrasound Fine-tuning Script
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add sam2 to path
sys.path.append(str(Path.cwd()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='sam2/configs/ultrasound_finetune.yaml')
    parser.add_argument('--device', default='0', help='GPU device ID')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Starting MedSAM2 ultrasound fine-tuning")
    print(f"   Config: {args.config}")
    print(f"   Device: {device}")
    
    # Import training modules
    from sam2.train import train_sam2
    
    # Start training
    train_sam2(
        config_file=args.config,
        device=device,
        resume_from=args.resume
    )

if __name__ == "__main__":
    main()
'''
    
    with open('train_ultrasound.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('train_ultrasound.py', 0o755)
    print("✅ Created training script: train_ultrasound.py")

def create_data_preparation_script():
    """Create script to prepare ultrasound data for training"""
    
    script_content = '''#!/usr/bin/env python3
"""
Prepare ultrasound data for MedSAM2 fine-tuning
Converts UNet format data to MedSAM2 training format
"""

import os
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
import argparse

def prepare_slice_data(volume_path, label_path, output_dir, slice_idx):
    """Prepare a single slice for training"""
    
    # Load volume and label
    vol_nib = nib.load(volume_path)
    lbl_nib = nib.load(label_path)
    
    vol_data = vol_nib.get_fdata().astype(np.float32)
    lbl_data = lbl_nib.get_fdata().astype(np.uint8)
    
    # Extract slice
    vol_slice = vol_data[:, :, slice_idx]
    lbl_slice = lbl_data[:, :, slice_idx]
    
    # Skip if no objects in label
    if lbl_slice.max() == 0:
        return None
    
    # Preprocess volume slice
    vol_slice = (vol_slice - vol_slice.min()) / (vol_slice.ptp() + 1e-6)
    vol_slice = cv2.resize(vol_slice, (512, 512), interpolation=cv2.INTER_LINEAR)
    vol_rgb = np.repeat(vol_slice[None], 3, axis=0)  # (3, 512, 512)
    
    # Preprocess label slice
    lbl_slice = cv2.resize(lbl_slice, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Generate prompts from label
    prompts = generate_prompts_from_label(lbl_slice)
    
    if prompts is None:
        return None
    
    # Save data
    base_name = f"{Path(volume_path).stem}_slice_{slice_idx:03d}"
    
    # Save image
    np.save(os.path.join(output_dir, 'images', f"{base_name}.npy"), vol_rgb)
    
    # Save prompts
    np.save(os.path.join(output_dir, 'prompts', f"{base_name}.npy"), prompts)
    
    # Save label
    np.save(os.path.join(output_dir, 'labels', f"{base_name}.npy"), lbl_slice)
    
    return base_name

def generate_prompts_from_label(label_512):
    """Generate prompts from label (mask+point strategy)"""
    
    # Find objects
    label_binary = (label_512 > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label_binary, connectivity=8)
    
    if num_labels <= 1:
        return None
    
    # Get largest object
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]
    
    if largest_area < 10:
        return None
    
    # Get centroid
    centroid_x = int(centroids[largest_label][0])
    centroid_y = int(centroids[largest_label][1])
    
    # Create prompts dictionary
    prompts = {
        'mask_input': label_512.astype(np.float32),
        'point_coords': np.array([[centroid_x, centroid_y]]),
        'point_labels': np.array([1])
    }
    
    return prompts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volumes', required=True, help='Directory with volume files')
    parser.add_argument('--labels', required=True, help='Directory with label files')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    args = parser.parse_args()
    
    # Create output directories
    output_dir = os.path.join(args.output, args.split)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'prompts'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Get volume and label files
    volume_files = sorted([f for f in os.listdir(args.volumes) if f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(args.labels) if f.endswith('.nii.gz')])
    
    print(f"📁 Processing {len(volume_files)} volumes...")
    
    total_slices = 0
    for vol_file, lbl_file in zip(volume_files, label_files):
        vol_path = os.path.join(args.volumes, vol_file)
        lbl_path = os.path.join(args.labels, lbl_file)
        
        # Load volume to get number of slices
        vol_nib = nib.load(vol_path)
        vol_data = vol_nib.get_fdata()
        num_slices = vol_data.shape[2]
        
        print(f"  Processing {vol_file} ({num_slices} slices)...")
        
        for slice_idx in range(num_slices):
            result = prepare_slice_data(vol_path, lbl_path, output_dir, slice_idx)
            if result:
                total_slices += 1
    
    print(f"✅ Prepared {total_slices} training slices in {output_dir}")

if __name__ == "__main__":
    main()
'''
    
    with open('prepare_ultrasound_data.py', 'w') as f:
        f.write(script_content)
    
    os.chmod('prepare_ultrasound_data.py', 0o755)
    print("✅ Created data preparation script: prepare_ultrasound_data.py")

def main():
    print("🔧 Setting up MedSAM2 ultrasound fine-tuning")
    print("=" * 50)
    
    # Create fine-tuning config
    config_path = create_finetune_config()
    
    # Create training script
    create_training_script()
    
    # Create data preparation script
    create_data_preparation_script()
    
    print("\n📋 Next steps:")
    print("1. Update paths in sam2/configs/ultrasound_finetune.yaml")
    print("2. Prepare your data:")
    print("   python prepare_ultrasound_data.py --volumes ./volumes --labels ./labels --output ./training_data --split train")
    print("   python prepare_ultrasound_data.py --volumes ./volumes --labels ./labels --output ./training_data --split val")
    print("3. Start fine-tuning:")
    print("   python train_ultrasound.py --device 0")
    
    print("\n💡 Key features:")
    print("- Uses MedSAM2_latest.pt as base weights")
    print("- mask+point prompt strategy (recommended)")
    print("- 512x512 input size (fixed)")
    print("- UNet format compatible")

if __name__ == "__main__":
    main() 