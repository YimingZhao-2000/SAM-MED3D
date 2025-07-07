#!/usr/bin/env python3
"""
Test script for ultrasound data loader
Tests the specific data format: date_id_part_numberPPM.nnrd and date_id_part_numberPPM_Mask.nnrd
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultrasound_data_loader import UltrasoundDataLoader

def test_data_loader():
    """Test the ultrasound data loader"""
    
    # Test data directory (you can modify this)
    test_data_dir = "ultrasound_data"
    
    if not os.path.exists(test_data_dir):
        print(f"❌ Test data directory not found: {test_data_dir}")
        print("Please create the directory and add your ultrasound data files:")
        print("  - date_id_part_numberPPM.nnrd (volume files)")
        print("  - date_id_part_numberPPM_Mask.nnrd (label files)")
        return False
    
    print(f"🔍 Testing data loader with directory: {test_data_dir}")
    
    try:
        # Initialize loader
        loader = UltrasoundDataLoader(test_data_dir)
        
        if not loader.data_files:
            print("❌ No data files found!")
            print("Expected files:")
            print("  - *PPM.nnrd (volume files)")
            print("  - *PPM_Mask.nnrd (label files)")
            return False
        
        print(f"✅ Found {len(loader.data_files)} data-label pairs")
        
        # Test loading first file
        if loader.data_files:
            data_file, label_file = loader.data_files[0]
            print(f"\n🧪 Testing file: {Path(data_file).name}")
            
            # Load data
            vol_data, lbl_data = loader.load_volume_and_label(data_file, label_file)
            
            if vol_data is None or lbl_data is None:
                print("❌ Failed to load data")
                return False
            
            # Validate data
            if not loader.validate_data(vol_data, lbl_data):
                print("❌ Data validation failed")
                return False
            
            # Print file info
            file_info = loader.get_file_info(data_file)
            print(f"✅ File info:")
            print(f"   Date: {file_info['date']}")
            print(f"   ID: {file_info['id']}")
            print(f"   Part: {file_info['part']}")
            print(f"   Number: {file_info['number']}")
            print(f"   Volume shape: {vol_data.shape}")
            print(f"   Label shape: {lbl_data.shape}")
            print(f"   Volume range: [{vol_data.min():.2f}, {vol_data.max():.2f}]")
            print(f"   Label range: [{lbl_data.min()}, {lbl_data.max()}]")
            print(f"   Label unique values: {set(lbl_data.flatten())}")
            
            return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def create_sample_data():
    """Create sample data for testing (if no real data available)"""
    
    import numpy as np
    import nibabel as nib
    
    test_dir = Path("ultrasound_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample volume and label
    shape = (256, 256, 32)
    
    # Sample volume (random ultrasound-like data)
    vol_data = np.random.rand(*shape).astype(np.float32)
    vol_data = (vol_data * 1000).astype(np.float32)  # Scale to typical ultrasound range
    
    # Sample label (binary mask)
    lbl_data = np.zeros(shape, dtype=np.uint8)
    
    # Add some random objects
    for i in range(3):
        # Random center
        cx = np.random.randint(50, 206)
        cy = np.random.randint(50, 206)
        cz = np.random.randint(5, 27)
        
        # Random size
        rx = np.random.randint(20, 40)
        ry = np.random.randint(20, 40)
        rz = np.random.randint(5, 10)
        
        # Create ellipsoid
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        mask = ((x - cx)**2 / rx**2 + (y - cy)**2 / ry**2 + (z - cz)**2 / rz**2) <= 1
        lbl_data[mask] = 1
    
    # Save sample files
    sample_name = "20240101_001_001_001PPM"
    
    vol_file = test_dir / f"{sample_name}.nnrd"
    lbl_file = test_dir / f"{sample_name}_Mask.nnrd"
    
    # Save as NIfTI files (nnrd format is typically NIfTI)
    vol_nib = nib.Nifti1Image(vol_data, affine=np.eye(4))
    lbl_nib = nib.Nifti1Image(lbl_data, affine=np.eye(4))
    
    nib.save(vol_nib, vol_file)
    nib.save(lbl_nib, lbl_file)
    
    print(f"✅ Created sample data:")
    print(f"   Volume: {vol_file}")
    print(f"   Label: {lbl_file}")
    print(f"   Shape: {vol_data.shape}")
    print(f"   Volume range: [{vol_data.min():.2f}, {vol_data.max():.2f}]")
    print(f"   Label unique: {set(lbl_data.flatten())}")

def main():
    """Main test function"""
    
    print("🧪 Testing Ultrasound Data Loader")
    print("=" * 40)
    
    # Check if test data exists
    test_data_dir = "ultrasound_data"
    
    if not os.path.exists(test_data_dir):
        print(f"📁 Test data directory not found: {test_data_dir}")
        print("Creating sample data for testing...")
        create_sample_data()
    
    # Run test
    success = test_data_loader()
    
    if success:
        print("\n🎉 All tests passed!")
        print("✅ Data loader is working correctly")
        print("✅ Ready to use with MedSAM2 inference")
    else:
        print("\n❌ Tests failed!")
        print("Please check your data format and try again")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 