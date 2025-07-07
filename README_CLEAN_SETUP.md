# MedSAM2 Ultrasound Inference - Clean Setup

Simple, robust inference scripts that don't require `chmod` and work reliably on servers.

## 🚀 Quick Start

### Option 1: Python Script (Recommended)
```bash
python3 run_inference.py
```

### Option 2: Bash Script
```bash
bash run_inference_clean.sh
```

### Option 3: Direct Command
```bash
python3 infer_medsam2_ultrasound.py \
    -i ./ultrasound_data \
    -o ./results \
    --prompt_mode mask+point \
    --data_structure us3d \
    --config_path sam2/configs \
    --yaml sam2.1_hiera_t512 \
    --device 0
```

## 📁 Data Structure

Place your ultrasound data in `ultrasound_data/`:
```
ultrasound_data/
├── volume1.nrrd                    # 3D volume
├── volume1_Mask.seg.nrrd          # 3D label
├── 20240101_123456_AB_01PPM.nrrd  # 3D volume
├── 20240101_123456_AB_01PPM_Mask.seg.nrrd  # 3D label
└── ...
```

## 📊 Output Structure

Results will be saved in:
```
results/
├── volume1_seg.nii.gz             # Segmentation result
├── 20240101_123456_AB_01PPM_seg.nii.gz
└── ...

logs/
├── inference_20240101_143022.log   # Inference log
└── ...

metrics/
├── metrics_20240101_143022.json   # Performance metrics
└── ...
```

## ✨ Features

### Clean Scripts
- ✅ **No chmod required** - Works immediately on any server
- ✅ **Robust error handling** - Clear error messages and graceful failures
- ✅ **Real-time output** - See progress as it happens
- ✅ **Automatic metrics** - Calculates Dice/IoU scores automatically
- ✅ **Colored output** - Easy to read status messages

### Smart Detection
- ✅ **Auto Python detection** - Uses `python3` or `python` automatically
- ✅ **File validation** - Checks data format before running
- ✅ **Directory creation** - Creates output dirs automatically
- ✅ **Timestamp logging** - Unique log files for each run

### Server-Friendly
- ✅ **No dependencies** - Pure Python/bash, no external tools
- ✅ **Error recovery** - Continues even if metrics fail
- ✅ **Resource efficient** - Minimal memory footprint
- ✅ **Logging** - Everything is logged for debugging

## 🔧 Configuration

### Environment Variables (Optional)
```bash
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export PYTHONPATH=.:$PYTHONPATH  # Add current dir to Python path
```

### Script Parameters
- `--device 0` - Use GPU 0 (change to `cpu` for CPU-only)
- `--prompt_mode mask+point` - Best for ultrasound data
- `--data_structure us3d` - Organized output structure

## 📈 Monitoring

### Real-time Monitoring
The scripts provide real-time feedback:
```
🔍 Checking requirements...
✅ Found 5 volume files and 5 mask files
📁 Creating output directories...
🚀 Starting MedSAM2 Inference
🔍 Running inference...
Processing volumes: 100%|██████████| 5/5 [02:30<00:00]
✅ Inference completed successfully!
📊 Calculating metrics...
✅ Metrics calculated successfully!
```

### Metrics Output
```json
{
  "total_files": 5,
  "processed_files": 5,
  "successful_files": 5,
  "overall_metrics": {
    "mean_dice_score": 0.8542,
    "std_dice_score": 0.0234,
    "mean_iou_score": 0.7432,
    "std_iou_score": 0.0312
  }
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **"No module named 'nibabel'"**
   ```bash
   pip install nibabel pynrrd opencv-python tqdm
   ```

2. **"Config file not found"**
   - Ensure `sam2/configs/` directory exists
   - Check that config files are present

3. **"CUDA out of memory"**
   ```bash
   # Use CPU instead
   python3 run_inference.py --device cpu
   ```

4. **"No data files found"**
   - Check that `.nrrd` files are in `ultrasound_data/`
   - Verify file naming: `*.nrrd` and `*_Mask.seg.nrrd`

### Debug Mode
For detailed debugging, run with verbose output:
```bash
python3 -u run_inference.py 2>&1 | tee debug.log
```

## 📝 Usage Examples

### Basic Inference
```bash
python3 run_inference.py
```

### Custom Parameters
```bash
python3 infer_medsam2_ultrasound.py \
    -i ./my_data \
    -o ./my_results \
    --prompt_mode point_only \
    --device cpu
```

### Monitor Only (after inference)
```bash
python3 monitor_inference.py \
    -i ./ultrasound_data \
    -o ./results \
    -r my_metrics.json
```

## 🎯 Best Practices

1. **Data Preparation**
   - Use `.nrrd` format for volumes and labels
   - Ensure consistent naming: `volume.nrrd` ↔ `volume_Mask.seg.nrrd`
   - Validate data before running inference

2. **Resource Management**
   - Use GPU for faster processing
   - Monitor memory usage with large datasets
   - Use CPU if GPU memory is insufficient

3. **Quality Control**
   - Check log files for errors
   - Review metrics for performance issues
   - Validate output segmentations

## 📞 Support

If you encounter issues:
1. Check the log files in `logs/`
2. Verify data format and naming
3. Ensure all dependencies are installed
4. Try running with `--device cpu` if GPU issues occur 