#!/usr/bin/env python3
"""
Clean MedSAM2 Ultrasound Inference Script
No shell scripts required, pure Python implementation
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

def print_colored(message, color="green"):
    """Print colored output"""
    colors = {
        "red": "\033[0;31m",
        "green": "\033[0;32m", 
        "yellow": "\033[1;33m",
        "blue": "\033[0;34m",
        "nc": "\033[0m"
    }
    print(f"{colors.get(color, '')}{message}{colors['nc']}")

def print_header(title):
    """Print section header"""
    print_colored("=" * 50, "blue")
    print_colored(title, "blue")
    print_colored("=" * 50, "blue")

def check_requirements():
    """Check if all requirements are met"""
    print_colored("🔍 Checking requirements...", "green")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print_colored("❌ Python 3.7+ required", "red")
        return False
    
    # Check input directory
    input_dir = Path("ultrasound_data")
    if not input_dir.exists():
        print_colored("❌ ultrasound_data/ directory not found", "red")
        print_colored("Please place your .nrrd files in ultrasound_data/", "yellow")
        return False
    
    # Count files
    volume_files = list(input_dir.glob("*.nrrd"))
    volume_files = [f for f in volume_files if not f.name.endswith("_Mask.seg.nrrd")]
    mask_files = list(input_dir.glob("*_Mask.seg.nrrd"))
    
    print_colored(f"✅ Found {len(volume_files)} volume files and {len(mask_files)} mask files", "green")
    
    if len(volume_files) == 0:
        print_colored("❌ No volume files found", "red")
        return False
    
    return True

def create_directories():
    """Create output directories"""
    print_colored("📁 Creating output directories...", "green")
    
    dirs = ["results", "logs", "metrics"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print_colored(f"✅ Created {dir_name}/", "green")

def run_inference():
    """Run the MedSAM2 inference"""
    print_header("🚀 Starting MedSAM2 Inference")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/inference_{timestamp}.log"
    metrics_file = f"metrics/metrics_{timestamp}.json"
    
    print_colored(f"📅 Timestamp: {timestamp}", "green")
    print_colored(f"📝 Log file: {log_file}", "green")
    print_colored(f"📊 Metrics file: {metrics_file}", "green")
    
    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "infer_medsam2_ultrasound.py",
        "-i", "ultrasound_data",
        "-o", "results", 
        "--prompt_mode", "mask+point",
        "--data_structure", "us3d",
        "--config_path", "sam2/configs",
        "--yaml", "sam2.1_hiera_t512",
        "--device", "0"
    ]
    
    print_colored("🔍 Running inference...", "green")
    print_colored(f"Command: {' '.join(cmd)}", "yellow")
    
    # Run inference
    try:
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line.rstrip())
                log_f.write(line)
                log_f.flush()
            
            return_code = process.wait()
            
        if return_code == 0:
            print_colored("✅ Inference completed successfully!", "green")
            return True, log_file, metrics_file
        else:
            print_colored(f"❌ Inference failed with return code {return_code}", "red")
            return False, log_file, metrics_file
            
    except Exception as e:
        print_colored(f"❌ Error running inference: {e}", "red")
        return False, log_file, metrics_file

def run_metrics(log_file, metrics_file):
    """Run metrics calculation"""
    print_colored("📊 Calculating metrics...", "green")
    
    if not Path("monitor_inference.py").exists():
        print_colored("⚠️  monitor_inference.py not found, skipping metrics", "yellow")
        return False
    
    cmd = [
        sys.executable,
        "monitor_inference.py",
        "-i", "ultrasound_data",
        "-o", "results", 
        "-r", metrics_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print_colored("✅ Metrics calculated successfully!", "green")
            return True
        else:
            print_colored("⚠️  Metrics calculation failed", "yellow")
            print_colored(result.stderr, "yellow")
            return False
    except Exception as e:
        print_colored(f"❌ Error calculating metrics: {e}", "red")
        return False

def print_summary(log_file, metrics_file):
    """Print inference summary"""
    print_header("📊 Inference Summary")
    
    # Count files
    input_dir = Path("ultrasound_data")
    output_dir = Path("results")
    
    volume_count = len([f for f in input_dir.glob("*.nrrd") if not f.name.endswith("_Mask.seg.nrrd")])
    output_count = len(list(output_dir.glob("*.nii.gz")))
    
    print_colored(f"📁 Input files: {volume_count}", "green")
    print_colored(f"📁 Output files: {output_count}", "green")
    print_colored(f"📝 Log file: {log_file}", "green")
    
    if Path(metrics_file).exists():
        print_colored(f"📊 Metrics file: {metrics_file}", "green")
    
    print_header("🎉 Success!")
    print_colored("📁 Results saved in: results/", "green")
    print_colored(f"📝 Log saved in: {log_file}", "green")
    if Path(metrics_file).exists():
        print_colored(f"📊 Metrics saved in: {metrics_file}", "green")

def main():
    """Main execution function"""
    print_header("MedSAM2 Ultrasound Inference")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run inference
    success, log_file, metrics_file = run_inference()
    
    if success:
        # Run metrics
        run_metrics(log_file, metrics_file)
        
        # Print summary
        print_summary(log_file, metrics_file)
    else:
        print_colored("❌ Inference failed! Check the log file for details.", "red")
        sys.exit(1)

if __name__ == "__main__":
    main() 