# MedSAM2 Ultrasound Inference - Minimal Setup

Simple, direct approach for 3D ultrasound segmentation using MedSAM2.

## Quick Start

### Zero-shot Inference

```bash
# 1. Setup environment
bash setup_ultrasound.sh

# 2. Test data loader (optional)
python test_data_loader.py

# 3. Prepare your ultrasound data
# Place your *.nnrd files in ultrasound_data/ directory:
# - date_id_part_numberPPM.nnrd (volume files)
# - date_id_part_numberPPM_Mask.nnrd (label files)

# 3. Run inference with ultrasound data
python infer_medsam2_ultrasound.py \
    -i ./ultrasound_data \
    -o ./results \
    --prompt_mode mask+point \
    --data_structure us3d
```

### Fine-tuning

```bash
# 1. Setup fine-tuning environment
python setup_finetune.py

# 2. Prepare training data
python prepare_ultrasound_data.py \
    --volumes ./data/volumes \
    --labels ./data/labels \
    --output ./training_data \
    --split train

# 3. Start fine-tuning
python train_ultrasound.py --device 0
```

## Data Format

### Ultrasound Data Format (Server)
```
📂ultrasound_data/
   ├── 20240101_001_001_001PPM.nnrd          # 3D volume (H×W×D, float32)
   ├── 20240101_001_001_001PPM_Mask.nnrd     # 3D label (H×W×D, uint8)
   ├── 20240101_001_001_002PPM.nnrd
   ├── 20240101_001_001_002PPM_Mask.nnrd
   └── ...
```

**File Naming Convention:**
- `date_id_part_numberPPM.nnrd` - Volume files
- `date_id_part_numberPPM_Mask.nnrd` - Label files
- Example: `20240101_001_001_001PPM.nnrd` and `20240101_001_001_001PPM_Mask.nnrd`

### Output Formats

#### UNet Format (Default)
```
📂results/
   ├── 20240101_001_001_001PPM.nii.gz        # Segmentation result
   ├── 20240101_001_001_002PPM.nii.gz
   └── ...
```

#### US3D Format (Organized)
```
📂results/
   ├── volumes/
   │   ├── 20240101_001_001_001PPM.nii.gz
   │   └── 20240101_001_001_002PPM.nii.gz
   ├── labels/
   │   ├── 20240101_001_001_001PPM.nii.gz
   │   └── 20240101_001_001_002PPM.nii.gz
   └── prompts/
       ├── 20240101_001_001_001PPM.nii.gz
       └── 20240101_001_001_002PPM.nii.gz
```

## Output

```
📂seg_ultrasound/
   ├── case_0001.nii.gz          # Binary segmentation (0/1)
   ├── case_0002.nii.gz
   └── ...
```

## Key Features

- **Slice-by-slice processing**: Each 2D slice → MedSAM2 → 3D reconstruction
- **Prompt support**: Auto center point, manual points, or no prompts
- **Preserves original dimensions**: Resizes back to original H×W
- **Batch processing**: Process entire directory at once

## Prompt Modes

MedSAM2 supports multiple types of prompts with intelligent filtering:

- **`mask+point`**: Combined mask and center of mass point (recommended)
- **`point_only`**: Center of mass point with variance filtering
- **`box_only`**: Bounding box with variance filtering
- **`mask`**: Exact object mask
- **`box`**: Bounding box around object
- **`auto`**: Uses center point as foreground prompt
- **`center`**: Manual center point prompt
- **`none`**: No prompts (may not work well)

### Ground Truth Prompts (Intelligent)

When you have ground truth labels, you can generate intelligent prompts:

- **`--prompt_from_gt`**: Enable ground truth-based prompts
- **`--gt_dir`**: Directory containing ground truth labels
- **`--gt_suffix`**: Suffix for ground truth files (default: `_gt.nii.gz`)

The script will automatically:
- Find corresponding ground truth files
- Extract object centroids, bounding boxes, or masks
- Generate appropriate prompts based on the selected mode

### Prompt Parameters

- **`--box_size`**: Size of center box (default: 100 pixels)
- **`--mask_radius`**: Radius of center mask (default: 50 pixels)

### Examples

```bash
# Point prompt with variance filtering
python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode point_only

# Box prompt with variance filtering
python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode box_only

# Combined mask and point (recommended)
python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode mask+point

# Test data loader
python test_data_loader.py

# Run inference with ultrasound data
python infer_medsam2_ultrasound.py -i ./ultrasound_data -o ./results --prompt_mode mask+point

# US3D data structure
python infer_medsam2_ultrasound.py -i ./ultrasound_data -o ./results --data_structure us3d --prompt_mode mask+point
```

## Tips

- Use `--device cpu` if GPU memory is insufficient
- For noisy ultrasound, consider adding preprocessing in `preprocess_slice()`
- Add prompts if needed: `prompts={'points': [[cx,cy]]}`

## Dependencies

- PyTorch
- nibabel (NIfTI I/O)
- opencv-python (image processing)
- tqdm (progress bars)
- MedSAM2 (model) 