#!/usr/bin/env python3
"""
Examples of different prompt types for MedSAM2 ultrasound inference
"""

import numpy as np
import cv2

def create_point_prompt(image_shape):
    """Create center point prompt"""
    H, W = image_shape[:2]
    point_coords = np.array([[W//2, H//2]])  # Center point
    point_labels = np.array([1])  # Foreground point
    return {
        'point_coords': point_coords,
        'point_labels': point_labels
    }

def create_box_prompt(image_shape, box_size=100):
    """Create center box prompt"""
    H, W = image_shape[:2]
    x1 = max(0, W//2 - box_size//2)
    y1 = max(0, H//2 - box_size//2)
    x2 = min(W, W//2 + box_size//2)
    y2 = min(H, H//2 + box_size//2)
    box = np.array([x1, y1, x2, y2])
    return {'box': box}

def create_mask_prompt(image_shape, radius=50):
    """Create center circular mask prompt"""
    H, W = image_shape[:2]
    center_x, center_y = W//2, H//2
    
    # Create circular mask
    y, x = np.ogrid[:H, :W]
    mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(np.float32)
    return {'mask_input': mask}

def create_combined_prompts(image_shape):
    """Create multiple prompt types for comparison"""
    H, W = image_shape[:2]
    
    # Point prompt
    point_coords = np.array([[W//2, H//2]])
    point_labels = np.array([1])
    
    # Box prompt
    box_size = 100
    x1 = max(0, W//2 - box_size//2)
    y1 = max(0, H//2 - box_size//2)
    x2 = min(W, W//2 + box_size//2)
    y2 = min(H, H//2 + box_size//2)
    box = np.array([x1, y1, x2, y2])
    
    # Mask prompt
    radius = 50
    y, x = np.ogrid[:H, :W]
    mask = ((x - W//2)**2 + (y - H//2)**2 <= radius**2).astype(np.float32)
    
    return {
        'point_coords': point_coords,
        'point_labels': point_labels,
        'box': box,
        'mask_input': mask
    }

def print_prompt_info(prompt_type, params):
    """Print information about the prompt"""
    print(f"\n📋 {prompt_type.upper()} PROMPT:")
    if 'point_coords' in params:
        print(f"  Point: {params['point_coords'][0]} (foreground)")
    if 'box' in params:
        print(f"  Box: {params['box']} [x1, y1, x2, y2]")
    if 'mask_input' in params:
        print(f"  Mask: Circular mask with radius {params.get('radius', 'N/A')}")

# Example usage
if __name__ == "__main__":
    print("🎯 MedSAM2 Prompt Examples")
    print("=" * 40)
    
    # Example image shape
    image_shape = (512, 512, 3)
    
    # Point prompt
    point_params = create_point_prompt(image_shape)
    print_prompt_info("point", point_params)
    
    # Box prompt
    box_params = create_box_prompt(image_shape, box_size=100)
    print_prompt_info("box", box_params)
    
    # Mask prompt
    mask_params = create_mask_prompt(image_shape, radius=50)
    print_prompt_info("mask", mask_params)
    
    # Combined prompts
    combined_params = create_combined_prompts(image_shape)
    print_prompt_info("combined", combined_params)
    
    print("\n💡 Usage in inference script:")
    print("python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode auto")
    print("python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode box --box_size 150")
    print("python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode mask --mask_radius 75") 