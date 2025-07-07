# SAM-MED3D: MedSAM2 Ultrasound Segmentation Workflow

<div align="center">

![MedSAM2 Ultrasound](https://img.shields.io/badge/MedSAM2-Ultrasound-00B89E?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Complete workflow for 3D ultrasound segmentation using MedSAM2 with UNet-format data**

</div>

## 🎯 Overview

SAM-MED3D provides a comprehensive solution for 3D ultrasound segmentation using MedSAM2, specifically designed for UNet-format medical data. This workflow includes both zero-shot inference and fine-tuning capabilities with intelligent prompt generation from ground truth labels.

## ✨ Key Features

- **🔬 Zero-shot Inference**: Use MedSAM2_latest.pt for immediate results
- **🎯 Intelligent Prompts**: Automatic prompt generation from ground truth labels
- **🔄 UNet Compatible**: Works with existing (D,H,W) NIfTI/.npy data
- **⚡ Multiple Prompt Types**: mask, point, box, and combined strategies
- **🚀 Fine-tuning Pipeline**: Complete training setup for custom models
- **📊 Production Ready**: Comprehensive testing and validation scripts

## 📋 Quick Start

### Environment Setup

```bash
# Create environment
conda create -n sam-med3d python=3.12 -y
conda activate sam-med3d

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

### Data Organization

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

### Zero-shot Inference

```bash
# Basic inference with ground truth prompts
python infer_medsam2_ultrasound.py \
    -i ./data/volumes \
    -o ./results \
    --prompt_from_gt \
    --gt_dir ./data/labels \
    --prompt_mode mask+point
```

### Fine-tuning

```bash
# Setup fine-tuning environment
python setup_finetune.py

# Prepare training data
python prepare_ultrasound_data.py \
    --volumes ./data/volumes \
    --labels ./data/labels \
    --output ./training_data \
    --split train

# Start fine-tuning
python train_ultrasound.py --device 0
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

## 🔧 Core Components

### Inference Scripts
- **`infer_medsam2_ultrasound.py`**: Main inference script with ground truth prompts
- **`test_inference.py`**: Validation script for all prompt types
- **`setup_ultrasound.sh`**: Environment setup script

### Fine-tuning Scripts
- **`setup_finetune.py`**: Creates fine-tuning configuration and scripts
- **`train_ultrasound.py`**: Training script for ultrasound fine-tuning
- **`prepare_ultrasound_data.py`**: Data preparation for training

### Configuration
- **`sam2/configs/ultrasound_finetune.yaml`**: Fine-tuning configuration
- **`README_ultrasound.md`**: Detailed usage instructions
- **`ULTRASOUND_WORKFLOW.md`**: Complete workflow guide

## 📈 Performance Tips

### Prompt Strategy Performance
- **mask+point**: Best accuracy, moderate speed
- **mask only**: Good accuracy, faster
- **box+point**: Good accuracy, faster
- **point only**: Fastest, lower accuracy

### Memory Management
- Use `batch_size=1` for training (single slice)
- Enable gradient checkpointing for large models
- Use mixed precision training (`torch.cuda.amp`)

## 🐛 Common Issues

### Input Size
**Problem**: Non-512×512 input  
**Solution**: Always resize to 512×512 (MedSAM2 requirement)

### Empty Prompts
**Problem**: No objects in label  
**Solution**: Skip slices with no objects or use fallback prompts

### Memory Issues
**Problem**: GPU out of memory  
**Solution**: Reduce batch size, use gradient checkpointing

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

## 📚 Documentation

- **[ULTRASOUND_WORKFLOW.md](ULTRASOUND_WORKFLOW.md)**: Complete workflow guide
- **[README_ultrasound.md](README_ultrasound.md)**: Detailed usage instructions
- **[examples/](examples/)**: Example scripts and data preparation

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MedSAM2](https://github.com/bowang-lab/MedSAM2) - Base model and architecture
- [SAM2](https://github.com/facebookresearch/sam2) - Original SAM2 implementation
- Medical imaging community for ultrasound datasets and feedback

---

<div align="center">

**SAM-MED3D: Advanced 3D Ultrasound Segmentation with MedSAM2**

[![GitHub stars](https://img.shields.io/github/stars/YimingZhao-2000/SAM-MED3D?style=social)](https://github.com/YimingZhao-2000/SAM-MED3D)
[![GitHub forks](https://img.shields.io/github/forks/YimingZhao-2000/SAM-MED3D?style=social)](https://github.com/YimingZhao-2000/SAM-MED3D)

</div>

