#!/usr/bin/env python3
"""
Debug script to check why data loader isn't finding files
"""

import os
from pathlib import Path

def debug_data_directory():
    """Debug the data directory to see what files exist"""
    
    data_dir = "ultrasound_data"
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory {data_dir} does not exist!")
        return
    
    print(f"🔍 Debugging directory: {data_dir}")
    print("=" * 50)
    
    # List all files
    all_files = list(Path(data_dir).glob("*"))
    print(f"📁 Total files found: {len(all_files)}")
    
    for file in all_files:
        print(f"  📄 {file.name}")
    
    print("\n🔍 Looking for .nrrd files:")
    nrrd_files = list(Path(data_dir).glob("*.nrrd"))
    print(f"  Found {len(nrrd_files)} .nrrd files:")
    for file in nrrd_files:
        print(f"    📄 {file.name}")
    
    print("\n🔍 Looking for _Mask.seg.nrrd files:")
    mask_files = list(Path(data_dir).glob("*_Mask.seg.nrrd"))
    print(f"  Found {len(mask_files)} _Mask.seg.nrrd files:")
    for file in mask_files:
        print(f"    📄 {file.name}")
    
    print("\n🔍 Checking for potential pairs:")
    for nrrd_file in nrrd_files:
        # Skip if it's already a mask file
        if nrrd_file.name.endswith("_Mask.seg.nrrd"):
            continue
            
        base_name = nrrd_file.stem
        expected_mask = nrrd_file.parent / f"{base_name}_Mask.seg.nrrd"
        
        print(f"  📄 Volume: {nrrd_file.name}")
        print(f"    Expected mask: {expected_mask.name}")
        print(f"    Mask exists: {expected_mask.exists()}")
        
        if expected_mask.exists():
            print(f"    ✅ PAIR FOUND!")
        else:
            print(f"    ❌ No matching mask found")
        print()

def test_file_patterns():
    """Test different file patterns"""
    
    print("\n🧪 Testing file patterns:")
    print("=" * 30)
    
    data_dir = Path("ultrasound_data")
    
    # Test different glob patterns
    patterns = [
        "*.nrrd",
        "*PPM.nrrd", 
        "*_Mask.seg.nrrd",
        "*_Mask.nrrd",
        "*.seg.nrrd",
        "*"
    ]
    
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        print(f"  Pattern '{pattern}': {len(files)} files")
        for file in files[:3]:  # Show first 3 files
            print(f"    - {file.name}")
        if len(files) > 3:
            print(f"    ... and {len(files) - 3} more")

if __name__ == "__main__":
    debug_data_directory()
    test_file_patterns() 