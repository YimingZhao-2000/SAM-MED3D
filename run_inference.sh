#!/bin/bash

# MedSAM2 Ultrasound Inference
# Simple and clean inference script

set -e  # Exit on error

echo "🚀 MedSAM2 Ultrasound Inference"
echo "=============================="

# Create directories
mkdir -p results logs metrics

# Check data
if [ ! -d "ultrasound_data" ]; then
    echo "❌ ultrasound_data/ not found"
    exit 1
fi

VOLUME_COUNT=$(find ultrasound_data/ -name "*.nrrd" ! -name "*_Mask.seg.nrrd" | wc -l)
echo "📊 Found ${VOLUME_COUNT} volume files"

if [ $VOLUME_COUNT -eq 0 ]; then
    echo "❌ No volume files found"
    exit 1
fi

# Run inference
echo "🔍 Starting inference..."
python infer_medsam2_ultrasound.py \
    -i ./ultrasound_data \
    -o ./results \
    --prompt_mode mask+point \
    --data_structure us3d \
    --config_path sam2/configs \
    --yaml sam2.1_hiera_t512 \
    --device 0

echo "✅ Inference completed!"
echo "📁 Results in: results/" 