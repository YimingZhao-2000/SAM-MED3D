#!/usr/bin/env python3
"""
Example showing how ground truth prompts work in MedSAM2 ultrasound inference
"""

import numpy as np
import cv2
import nibabel as nib
import os

def create_dummy_gt_mask(shape, num_objects=2):
    """Create a dummy ground truth mask for demonstration"""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Create some dummy objects
    for i in range(num_objects):
        # Random position and size
        center_x = np.random.randint(50, W-50)
        center_y = np.random.randint(50, H-50)
        radius = np.random.randint(20, 40)
        
        # Create circular object
        y, x = np.ogrid[:H, :W]
        circle = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
        mask[circle] = i + 1
    
    return mask

def demonstrate_gt_prompts():
    """Demonstrate different ground truth prompt types"""
    
    print("🎯 Ground Truth Prompt Examples")
    print("=" * 50)
    
    # Create dummy ground truth
    H, W = 256, 256
    gt_mask = create_dummy_gt_mask((H, W), num_objects=2)
    
    print(f"Created dummy ground truth mask: {gt_mask.shape}")
    print(f"Objects found: {len(np.unique(gt_mask)) - 1}")
    
    # Find connected components
    gt_binary = (gt_mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_binary, connectivity=8)
    
    print(f"\nConnected components analysis:")
    print(f"  Total components: {num_labels}")
    
    # Analyze each object
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        centroid_x = int(centroids[i][0])
        centroid_y = int(centroids[i][1])
        
        print(f"\n  Object {i}:")
        print(f"    Area: {area} pixels")
        print(f"    Centroid: ({centroid_x}, {centroid_y})")
        print(f"    Bounding box: [{x}, {y}, {x+w}, {y+h}]")
        
        # Show different prompt types
        print(f"    Prompts:")
        print(f"      Point: [{centroid_x}, {centroid_y}] (foreground)")
        print(f"      Box: [{x}, {y}, {x+w}, {y+h}]")
        print(f"      Mask: Binary mask of object {i}")
    
    return gt_mask

def show_file_matching():
    """Show how ground truth files are matched"""
    
    print("\n📁 Ground Truth File Matching")
    print("=" * 40)
    
    # Example input files
    input_files = [
        "case_001.nii.gz",
        "patient_123.nii.gz", 
        "ultrasound_scan.nii.gz"
    ]
    
    # Different ground truth naming patterns
    gt_patterns = [
        "_gt.nii.gz",
        "_label.nii.gz", 
        "_mask.nii.gz",
        "_seg.nii.gz"
    ]
    
    print("Input file → Possible ground truth files:")
    for input_file in input_files:
        base_name = input_file.replace('.nii.gz', '')
        print(f"\n  {input_file}:")
        for pattern in gt_patterns:
            gt_file = f"{base_name}{pattern}"
            print(f"    → {gt_file}")
        # Also show reverse pattern
        print(f"    → gt_{base_name}.nii.gz")

def usage_examples():
    """Show usage examples"""
    
    print("\n💡 Usage Examples")
    print("=" * 30)
    
    examples = [
        {
            "description": "Basic ground truth prompts (point)",
            "command": "python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_from_gt --gt_dir ./ground_truth --prompt_mode auto"
        },
        {
            "description": "Ground truth box prompts",
            "command": "python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_from_gt --gt_dir ./ground_truth --prompt_mode box"
        },
        {
            "description": "Ground truth mask prompts",
            "command": "python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_from_gt --gt_dir ./ground_truth --prompt_mode mask"
        },
        {
            "description": "Custom ground truth suffix",
            "command": "python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_from_gt --gt_dir ./ground_truth --gt_suffix _label.nii.gz --prompt_mode auto"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}")
        print(f"   {example['command']}")

if __name__ == "__main__":
    # Demonstrate ground truth prompt generation
    gt_mask = demonstrate_gt_prompts()
    
    # Show file matching
    show_file_matching()
    
    # Show usage examples
    usage_examples()
    
    print("\n✅ Ground truth prompts provide intelligent, object-aware prompts!")
    print("   Instead of center points, you get prompts at actual object locations.") 