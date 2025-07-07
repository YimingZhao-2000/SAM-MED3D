#!/bin/bash
# Minimal setup for MedSAM2 ultrasound inference
# Follows Occam's Razor - simple and direct

echo "🔧 Minimal MedSAM2 Setup for Ultrasound"

# 1. Create environment
conda create -n medsam2 python=3.12 -y
conda activate medsam2

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision

# 3. Install MedSAM2
pip install -e ".[dev]"

# 4. Install additional dependencies for ultrasound inference
pip install nibabel opencv-python tqdm

# 5. Check if checkpoints exist
if [ ! -f "checkpoints/MedSAM2_latest.pt" ]; then
    echo "📥 Downloading latest weights..."
    wget -P checkpoints \
        https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt
else
    echo "✅ Weights already exist"
fi

echo "✅ Setup complete!"
echo ""
echo "Testing installation..."
python3 test_inference.py
echo ""
echo "Usage:"
echo "python infer_medsam2_ultrasound.py -i ./data_ultrasound -o ./seg_ultrasound --prompt_mode auto" 