# MedSAM2 Ultrasound Inference - Minimal Setup

Simple, direct approach for 3D ultrasound segmentation using MedSAM2.

## Quick Start

### Zero-shot Inference

```bash
# 1. Setup environment
bash setup_ultrasound.sh

# 2. Prepare data (UNet format)
mkdir -p data/volumes data/labels
# Copy your *.nii.gz volumes to data/volumes/
# Copy your *.nii.gz labels to data/labels/

# 3. Run inference with ground truth prompts
python infer_medsam2_ultrasound.py \
    -i ./data/volumes \
    -o ./results \
    --prompt_from_gt \
    --gt_dir ./data/labels \
    --prompt_mode mask+point
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

```
📂data_ultrasound/
   ├── case_0001.nii.gz          # 3D volume (H×W×D, float32)
   ├── case_0002.nii.gz
   └── ...
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

MedSAM2 supports multiple types of prompts:

- **`auto`**: Uses center point as foreground prompt (recommended)
- **`center`**: Manual center point prompt
- **`box`**: Uses center bounding box prompt
- **`mask`**: Uses center circular mask prompt
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
# Point prompt (default)
python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode auto

# Box prompt
python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode box --box_size 150

# Mask prompt
python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode mask --mask_radius 75

# Ground truth prompts (intelligent)
python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_from_gt --gt_dir ./ground_truth --prompt_mode auto
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