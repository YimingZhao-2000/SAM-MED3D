#!/bin/bash

# MedSAM2 Ultrasound Inference with Monitoring
# This script runs inference and calculates loss metrics

echo "🚀 Starting MedSAM2 Ultrasound Inference with Monitoring"
echo "=================================================="

# Create output directories
mkdir -p results
mkdir -p logs
mkdir -p metrics

# Set timestamp for logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/inference_${TIMESTAMP}.log"
METRICS_FILE="metrics/inference_metrics_${TIMESTAMP}.json"

echo "📁 Input directory: ultrasound_data/"
echo "📁 Output directory: results/"
echo "📁 Log file: ${LOG_FILE}"
echo "📁 Metrics file: ${METRICS_FILE}"

# Check if data directory exists
if [ ! -d "ultrasound_data" ]; then
    echo "❌ Error: ultrasound_data/ directory not found!"
    echo "Please ensure your .nrrd files are in ultrasound_data/"
    exit 1
fi

# Count input files
VOLUME_COUNT=$(find ultrasound_data/ -name "*.nrrd" ! -name "*_Mask.seg.nrrd" | wc -l)
MASK_COUNT=$(find ultrasound_data/ -name "*_Mask.seg.nrrd" | wc -l)

echo "📊 Found ${VOLUME_COUNT} volume files and ${MASK_COUNT} mask files"

if [ $VOLUME_COUNT -eq 0 ]; then
    echo "❌ No volume files found in ultrasound_data/"
    exit 1
fi

# Run inference with monitoring
echo "🔍 Starting inference..."
python infer_medsam2_ultrasound.py \
    -i ./ultrasound_data \
    -o ./results \
    --prompt_mode mask+point \
    --data_structure us3d \
    --calculate_loss \
    --save_metrics \
    --log_file ${LOG_FILE} \
    --metrics_file ${METRICS_FILE} \
    --monitor_progress \
    --device 0 2>&1 | tee ${LOG_FILE}

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo "✅ Inference completed successfully!"
    
    # Display summary
    echo ""
    echo "📊 Inference Summary:"
    echo "===================="
    echo "Input files: ${VOLUME_COUNT}"
    echo "Output files: $(find results/ -name "*.nii.gz" | wc -l)"
    echo "Log file: ${LOG_FILE}"
    echo "Metrics file: ${METRICS_FILE}"
    
    # Show metrics if available
    if [ -f "${METRICS_FILE}" ]; then
        echo ""
        echo "📈 Metrics Summary:"
        echo "=================="
        python -c "
import json
import sys
try:
    with open('${METRICS_FILE}', 'r') as f:
        metrics = json.load(f)
    print(f'Dice Score: {metrics.get(\"dice_score\", \"N/A\"):.4f}')
    print(f'Hausdorff Distance: {metrics.get(\"hausdorff_distance\", \"N/A\"):.4f}')
    print(f'IoU Score: {metrics.get(\"iou_score\", \"N/A\"):.4f}')
    print(f'Processing Time: {metrics.get(\"processing_time\", \"N/A\"):.2f}s')
except Exception as e:
    print(f'Could not read metrics: {e}')
"
    fi
    
else
    echo "❌ Inference failed! Check the log file: ${LOG_FILE}"
    exit 1
fi

echo ""
echo "🎉 Inference pipeline completed!"
echo "📁 Results saved in: results/"
echo "📊 Metrics saved in: ${METRICS_FILE}"
echo "📝 Log saved in: ${LOG_FILE}" 