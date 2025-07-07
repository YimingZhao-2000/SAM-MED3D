#!/usr/bin/env python3
"""
Ultrasound Data Loader for SAM-MED3D
Handles specific data format: date_id_part_numberPPM.nnrd and date_id_part_numberPPM_Mask.nnrd
"""

import os
import re
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, List, Optional
import argparse

class UltrasoundDataLoader:
    """Loader for ultrasound data with specific naming convention"""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing ultrasound data files
        """
        self.data_dir = Path(data_dir)
        self.data_files = self._find_data_files()
        
    def _find_data_files(self) -> List[Tuple[str, str]]:
        """
        Find data files and their corresponding labels
        
        Returns:
            List of tuples (data_file, label_file)
        """
        data_files = []
        
        # Find all .nrrd files (data files)
        volume_files = list(self.data_dir.glob("*.nrrd"))
        
        for volume_file in volume_files:
            # Skip mask files (they end with _Mask.seg.nrrd)
            if volume_file.name.endswith("_Mask.seg.nrrd"):
                continue
                
            # Extract base name (everything before .nrrd)
            base_name = volume_file.stem
            
            # Look for corresponding mask file (with _Mask.seg.nrrd extension)
            mask_file = volume_file.parent / f"{base_name}_Mask.seg.nrrd"
            
            if mask_file.exists():
                data_files.append((str(volume_file), str(mask_file)))
                print(f"✅ Found pair: {volume_file.name} ↔ {mask_file.name}")
            else:
                print(f"⚠️  No mask found for: {volume_file.name}")
                print(f"   Expected mask file: {mask_file.name}")
        
        print(f"📁 Found {len(data_files)} data-label pairs")
        return data_files
    
    def load_volume_and_label(self, data_file: str, label_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load volume and label from files
        
        Args:
            data_file: Path to volume file
            label_file: Path to label file
            
        Returns:
            Tuple of (volume_data, label_data)
        """
        try:
            # Load volume using nrrd
            import nrrd
            vol_data, vol_header = nrrd.read(data_file)
            vol_data = vol_data.astype(np.float32)
            
            # Load label using nrrd
            lbl_data, lbl_header = nrrd.read(label_file)
            lbl_data = lbl_data.astype(np.uint8)
            
            # Ensure same dimensions
            if vol_data.shape != lbl_data.shape:
                print(f"⚠️  Shape mismatch: volume {vol_data.shape} vs label {lbl_data.shape}")
                # Resize label to match volume
                lbl_data = self._resize_label_to_volume(lbl_data, vol_data.shape)
            
            return vol_data, lbl_data
            
        except Exception as e:
            print(f"❌ Error loading {data_file}: {e}")
            return None, None
    
    def _resize_label_to_volume(self, label: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Resize label to match volume dimensions"""
        import cv2
        
        if len(label.shape) == 3:
            # 3D volume - resize slice by slice
            resized_label = np.zeros(target_shape, dtype=label.dtype)
            for z in range(min(label.shape[2], target_shape[2])):
                slice_2d = label[:, :, z]
                resized_slice = cv2.resize(slice_2d, (target_shape[1], target_shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                resized_label[:, :, z] = resized_slice
            return resized_label
        else:
            # 2D slice
            return cv2.resize(label, (target_shape[1], target_shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
    
    def get_file_info(self, data_file: str) -> dict:
        """
        Extract information from filename
        
        Args:
            data_file: Path to data file
            
        Returns:
            Dictionary with parsed information
        """
        filename = Path(data_file).name
        
        # Parse any .nrrd file (flexible naming)
        # Extract base name for any file ending with .nrrd
        base_name = filename.replace('.nrrd', '')
        
        # Try to parse structured naming if it matches the pattern
        pattern = r'(\d{8})_(\d{6})_([A-Z]{2,3})_(\d{2})PPM'
        match = re.match(pattern, base_name)
        
        if match:
            date, id_6digits, id_2to3chars, number = match.groups()
            return {
                'date': date,
                'id_6digits': id_6digits,
                'id_2to3chars': id_2to3chars,
                'number': number,
                'filename': filename,
                'base_name': base_name
            }
        else:
            # Fallback for any naming pattern
            return {
                'filename': filename,
                'base_name': base_name,
                'date': 'unknown',
                'id_6digits': 'unknown',
                'id_2to3chars': 'unknown',
                'number': 'unknown'
            }
    
    def validate_data(self, vol_data: np.ndarray, lbl_data: np.ndarray) -> bool:
        """
        Validate loaded data
        
        Args:
            vol_data: Volume data
            lbl_data: Label data
            
        Returns:
            True if valid, False otherwise
        """
        if vol_data is None or lbl_data is None:
            return False
        
        # Check shapes
        if vol_data.shape != lbl_data.shape:
            print(f"❌ Shape mismatch: {vol_data.shape} vs {lbl_data.shape}")
            return False
        
        # Check data ranges
        if vol_data.min() < 0 or vol_data.max() > 1e6:
            print(f"⚠️  Unusual volume range: [{vol_data.min():.2f}, {vol_data.max():.2f}]")
        
        if lbl_data.min() < 0 or lbl_data.max() > 10:
            print(f"⚠️  Unusual label range: [{lbl_data.min()}, {lbl_data.max()}]")
        
        # Check for empty labels
        if lbl_data.max() == 0:
            print(f"⚠️  Empty label data")
            return False
        
        return True
    
    def list_all_files(self) -> List[dict]:
        """
        List all data files with their information
        
        Returns:
            List of dictionaries with file information
        """
        file_info_list = []
        
        for data_file, label_file in self.data_files:
            info = self.get_file_info(data_file)
            info['data_file'] = data_file
            info['label_file'] = label_file
            file_info_list.append(info)
        
        return file_info_list

def main():
    """Test the data loader"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing ultrasound data')
    parser.add_argument('--test_load', action='store_true', help='Test loading a sample file')
    args = parser.parse_args()
    
    # Initialize loader
    loader = UltrasoundDataLoader(args.data_dir)
    
    # List all files
    print("\n📋 Data Files Found:")
    print("=" * 50)
    for info in loader.list_all_files():
        print(f"📁 {info['filename']}")
        print(f"   Date: {info['date']}, ID: {info['id']}, Part: {info['part']}, Number: {info['number']}")
        print(f"   Data: {info['data_file']}")
        print(f"   Label: {info['label_file']}")
        print()
    
    # Test loading if requested
    if args.test_load and loader.data_files:
        print("\n🧪 Testing Data Loading:")
        print("=" * 30)
        
        data_file, label_file = loader.data_files[0]
        vol_data, lbl_data = loader.load_volume_and_label(data_file, label_file)
        
        if loader.validate_data(vol_data, lbl_data):
            print(f"✅ Successfully loaded: {Path(data_file).name}")
            print(f"   Volume shape: {vol_data.shape}")
            print(f"   Label shape: {lbl_data.shape}")
            print(f"   Volume range: [{vol_data.min():.2f}, {vol_data.max():.2f}]")
            print(f"   Label range: [{lbl_data.min()}, {lbl_data.max()}]")
            print(f"   Label unique values: {np.unique(lbl_data)}")
        else:
            print(f"❌ Failed to load: {Path(data_file).name}")

if __name__ == "__main__":
    main() 