#!/usr/bin/env bash

# MedSAM2 Ultrasound Inference - Clean Version
# No chmod required, robust for server use

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Main execution
main() {
    print_header "MedSAM2 Ultrasound Inference"
    
    # Configuration
    INPUT_DIR="ultrasound_data"
    OUTPUT_DIR="results"
    LOG_DIR="logs"
    METRICS_DIR="metrics"
    
    # Create directories
    print_status "Creating output directories..."
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$METRICS_DIR"
    
    # Generate timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/inference_${TIMESTAMP}.log"
    METRICS_FILE="$METRICS_DIR/metrics_${TIMESTAMP}.json"
    
    print_status "Timestamp: $TIMESTAMP"
    print_status "Log file: $LOG_FILE"
    print_status "Metrics file: $METRICS_FILE"
    
    # Check input directory
    if [[ ! -d "$INPUT_DIR" ]]; then
        print_error "Input directory '$INPUT_DIR' not found!"
        print_status "Please place your .nrrd files in the '$INPUT_DIR' directory"
        exit 1
    fi
    
    # Count files
    VOLUME_COUNT=$(find "$INPUT_DIR" -name "*.nrrd" ! -name "*_Mask.seg.nrrd" 2>/dev/null | wc -l)
    MASK_COUNT=$(find "$INPUT_DIR" -name "*_Mask.seg.nrrd" 2>/dev/null | wc -l)
    
    print_status "Found $VOLUME_COUNT volume files and $MASK_COUNT mask files"
    
    if [[ $VOLUME_COUNT -eq 0 ]]; then
        print_error "No volume files found in '$INPUT_DIR'"
        print_status "Expected files: *.nrrd (volumes) and *_Mask.seg.nrrd (labels)"
        exit 1
    fi
    
    # Check Python availability
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found! Please install Python 3.7+"
        exit 1
    fi
    
    print_status "Using Python: $PYTHON_CMD"
    
    # Run inference
    print_header "Starting Inference"
    print_status "Running MedSAM2 inference..."
    
    if $PYTHON_CMD infer_medsam2_ultrasound.py \
        -i "$INPUT_DIR" \
        -o "$OUTPUT_DIR" \
        --prompt_mode mask+point \
        --data_structure us3d \
        --config_path sam2/configs \
        --yaml sam2.1_hiera_t512 \
        --device 0 2>&1 | tee "$LOG_FILE"; then
        
        print_status "Inference completed successfully!"
        
        # Calculate metrics if monitoring script exists
        if [[ -f "monitor_inference.py" ]]; then
            print_status "Calculating metrics..."
            if $PYTHON_CMD monitor_inference.py -i "$INPUT_DIR" -o "$OUTPUT_DIR" -r "$METRICS_FILE"; then
                print_status "Metrics calculated successfully!"
            else
                print_warning "Metrics calculation failed, but inference was successful"
            fi
        else
            print_warning "monitor_inference.py not found, skipping metrics calculation"
        fi
        
        # Summary
        print_header "Inference Summary"
        OUTPUT_COUNT=$(find "$OUTPUT_DIR" -name "*.nii.gz" 2>/dev/null | wc -l)
        print_status "Input files: $VOLUME_COUNT"
        print_status "Output files: $OUTPUT_COUNT"
        print_status "Log file: $LOG_FILE"
        if [[ -f "$METRICS_FILE" ]]; then
            print_status "Metrics file: $METRICS_FILE"
        fi
        
        print_header "Success!"
        print_status "Results saved in: $OUTPUT_DIR/"
        print_status "Log saved in: $LOG_FILE"
        if [[ -f "$METRICS_FILE" ]]; then
            print_status "Metrics saved in: $METRICS_FILE"
        fi
        
    else
        print_error "Inference failed!"
        print_status "Check the log file for details: $LOG_FILE"
        exit 1
    fi
}

# Run main function
main "$@" 