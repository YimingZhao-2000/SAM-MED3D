# Ultrasound Data Directory

Place your 3D ultrasound volumes here in NIfTI format.

## Expected Structure

```
📂examples/ultrasound_data/
   ├── case_0001.nii.gz          # 3D volume (H×W×D, float32)
   ├── case_0002.nii.gz
   └── ...
```

## Usage

```bash
# Run inference on this directory
python ../infer_medsam2_ultrasound.py \
    -i ./examples/ultrasound_data \
    -o ./examples/ultrasound_results
```

## Data Requirements

- **Format**: NIfTI (.nii or .nii.gz)
- **Data type**: float32
- **Dimensions**: (Height, Width, Depth)
- **Values**: Normalized ultrasound intensities 