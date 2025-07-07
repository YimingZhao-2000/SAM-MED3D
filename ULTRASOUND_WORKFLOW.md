# MedSAM2 Ultrasound Workflow - Complete Guide

## 🎯 Overview

This guide provides a complete workflow for using MedSAM2 with UNet-format 3D ultrasound data, including both inference and fine-tuning.

## 📋 Key Principles

- **Weights**: Use only `MedSAM2_latest.pt`
- **Prompts**: Generate from ground truth labels (mask/box/point), **never use empty prompts**
- **Data**: Compatible with existing `(D,H,W)` NIfTI/.npy volumes + corresponding labels
- **Input Size**: Fixed 512×512 (MedSAM2 requirement)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create environment
conda create -n medsam2 python=3.12 -y
conda activate medsam2

# Install PyTorch (adjust CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install MedSAM2
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2
pip install -e ".[dev]"

# Download latest weights
mkdir -p checkpoints
wget -P checkpoints \
  https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt
```

### 2. Data Preparation

Your data should be organized as:
```
📁 data/
├── volumes/
│   ├── case_001.nii.gz    # (H,W,D) float32, 0-1 normalized
│   ├── case_002.nii.gz
│   └── ...
└── labels/
    ├── case_001.nii.gz    # (H,W,D) uint8, binary labels
    ├── case_002.nii.gz
    └── ...
```

### 3. Inference (Zero-shot)

```bash
# Basic inference with ground truth prompts
python infer_medsam2_ultrasound.py \
    -i ./data/volumes \
    -o ./results \
    --prompt_from_gt \
    --gt_dir ./data/labels \
    --prompt_mode mask+point

# Different prompt strategies
python infer_medsam2_ultrasound.py \
    -i ./data/volumes \
    -o ./results \
    --prompt_from_gt \
    --gt_dir ./data/labels \
    --prompt_mode mask      # mask only

python infer_medsam2_ultrasound.py \
    -i ./data/volumes \
    -o ./results \
    --prompt_from_gt \
    --gt_dir ./data/labels \
    --prompt_mode box       # bounding box
```

## 🔧 Fine-tuning Setup

### 1. Setup Fine-tuning Environment

```bash
# Create fine-tuning scripts and configs
python setup_finetune.py
```

This creates:
- `sam2/configs/ultrasound_finetune.yaml` - Fine-tuning configuration
- `train_ultrasound.py` - Training script
- `prepare_ultrasound_data.py` - Data preparation script

### 2. Prepare Training Data

```bash
# Prepare training data
python prepare_ultrasound_data.py \
    --volumes ./data/volumes \
    --labels ./data/labels \
    --output ./training_data \
    --split train

python prepare_ultrasound_data.py \
    --volumes ./data/volumes \
    --labels ./data/labels \
    --output ./training_data \
    --split val
```

### 3. Update Configuration

Edit `sam2/configs/ultrasound_finetune.yaml`:
```yaml
dataset:
  train_root: /path/to/training_data/train
  train_label_root: /path/to/training_data/train
  val_root: /path/to/training_data/val
  val_label_root: /path/to/training_data/val
  modality: ultrasound
  prompt_type: mask+point
```

### 4. Start Fine-tuning

```bash
# Single GPU training
python train_ultrasound.py --device 0

# Multi-GPU training
torchrun --nproc_per_node=2 train_ultrasound.py --device 0,1
```

## 📊 Prompt Strategies

### 1. Mask + Point (Recommended)
```python
prompts = {
    'mask_input': label_512.astype(np.float32),  # Exact object mask
    'point_coords': np.array([[cx, cy]]),        # Object centroid
    'point_labels': np.array([1])                # Foreground point
}
```

### 2. Mask Only
```python
prompts = {
    'mask_input': label_512.astype(np.float32)   # Exact object mask
}
```

### 3. Box + Point
```python
prompts = {
    'box': np.array([x1, y1, x2, y2]),          # Bounding box
    'point_coords': np.array([[cx, cy]]),        # Object centroid
    'point_labels': np.array([1])                # Foreground point
}
```

## 🔄 Data Processing Pipeline

### Volume Processing
1. **Load**: `(H,W,D)` float32 NIfTI volume
2. **Normalize**: 0-1 range
3. **Slice**: Process each D slice independently
4. **Resize**: `cv2.resize` to 512×512
5. **Convert**: `np.repeat` to RGB (3,512,512)

### Label Processing
1. **Load**: `(H,W,D)` uint8 NIfTI label
2. **Slice**: Extract corresponding slice
3. **Resize**: `cv2.resize` to 512×512 (nearest neighbor)
4. **Analyze**: Find connected components
5. **Extract**: Generate prompts (centroid, bounding box, mask)

## 📈 Performance Tips

### 1. Prompt Strategy
- **mask+point**: Best accuracy, moderate speed
- **mask only**: Good accuracy, faster
- **box+point**: Good accuracy, faster
- **point only**: Fastest, lower accuracy

### 2. Memory Management
- Use `batch_size=1` for training (single slice)
- Enable gradient checkpointing for large models
- Use mixed precision training (`torch.cuda.amp`)

### 3. Data Augmentation
- Random rotation (±15°)
- Random scaling (0.8-1.2)
- Random brightness/contrast
- Random noise injection

## 🐛 Common Issues

### 1. Input Size
**Problem**: Non-512×512 input
**Solution**: Always resize to 512×512 (MedSAM2 requirement)

### 2. Empty Prompts
**Problem**: No objects in label
**Solution**: Skip slices with no objects or use fallback prompts

### 3. Memory Issues
**Problem**: GPU out of memory
**Solution**: Reduce batch size, use gradient checkpointing

### 4. File Matching
**Problem**: Can't find ground truth files
**Solution**: Check naming convention, use `--gt_suffix` parameter

## 📝 Example Workflow

```bash
# 1. Setup environment
bash setup_ultrasound.sh

# 2. Test inference
python test_inference.py

# 3. Run zero-shot inference
python infer_medsam2_ultrasound.py \
    -i ./examples/ultrasound_data \
    -o ./examples/ultrasound_results \
    --prompt_from_gt \
    --gt_dir ./examples/ultrasound_labels \
    --prompt_mode mask+point

# 4. Setup fine-tuning
python setup_finetune.py

# 5. Prepare training data
python prepare_ultrasound_data.py \
    --volumes ./examples/ultrasound_data \
    --labels ./examples/ultrasound_labels \
    --output ./training_data \
    --split train

# 6. Start fine-tuning
python train_ultrasound.py --device 0
```

## 🎯 Key Advantages

1. **No Re-annotation**: Use existing UNet labels directly
2. **Intelligent Prompts**: Automatic prompt generation from ground truth
3. **Flexible Strategy**: Multiple prompt types for different scenarios
4. **Production Ready**: Complete inference and training pipeline
5. **UNet Compatible**: Works with existing ultrasound data formats

This workflow provides a complete solution for MedSAM2 ultrasound segmentation, from zero-shot inference to fine-tuned models! 