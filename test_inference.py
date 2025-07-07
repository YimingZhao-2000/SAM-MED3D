#!/usr/bin/env python3
"""
Test script to verify MedSAM2 inference works correctly
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2 imports successful")
    except ImportError as e:
        print(f"❌ SAM2 import failed: {e}")
        return False
    
    try:
        import nibabel as nib
        print("✅ nibabel import successful")
    except ImportError as e:
        print(f"❌ nibabel import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv import successful")
    except ImportError as e:
        print(f"❌ opencv import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if model loads correctly"""
    print("\nTesting model loading...")
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Check if checkpoint exists
        checkpoint_path = "checkpoints/MedSAM2_latest.pt"
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        # Load model
        sam2_model = build_sam2(
            config_file="sam2/configs/sam2.1_hiera_t512.yaml",
            ckpt_path=checkpoint_path,
            device="cpu",  # Use CPU for testing
            mode="eval"
        )
        
        # Create predictor
        predictor = SAM2ImagePredictor(sam2_model)
        print("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_inference_workflow():
    """Test the complete inference workflow"""
    print("\nTesting inference workflow...")
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import cv2
        
        # Load model
        sam2_model = build_sam2(
            config_file="sam2/configs/sam2.1_hiera_t512.yaml",
            ckpt_path="checkpoints/MedSAM2_latest.pt",
            device="cpu",
            mode="eval"
        )
        predictor = SAM2ImagePredictor(sam2_model)
        
        # Create a dummy 2D slice
        dummy_slice = np.random.rand(256, 256).astype(np.float32)
        
        # Preprocess
        img2d = (dummy_slice - dummy_slice.min()) / (dummy_slice.ptp() + 1e-6) * 255
        img2d = img2d.astype(np.uint8)
        img2d = cv2.resize(img2d, (512, 512), interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([img2d, img2d, img2d], axis=2)  # (H,W,3)
        
        # Set image
        predictor.set_image(rgb)
        
        # Test different prompt types
        print("Testing point prompts...")
        point_coords = np.array([[256, 256]])  # Center point
        point_labels = np.array([1])  # Foreground point
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )
        print(f"  ✅ Point prompts: {len(masks)} masks")
        
        print("Testing box prompts...")
        box = np.array([200, 200, 312, 312])  # Center box
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=False
        )
        print(f"  ✅ Box prompts: {len(masks)} masks")
        
        print("Testing mask prompts...")
        # Create circular mask
        y, x = np.ogrid[:512, :512]
        mask = ((x - 256)**2 + (y - 256)**2 <= 50**2).astype(np.float32)
        masks, scores, logits = predictor.predict(
            mask_input=mask,
            multimask_output=False
        )
        print(f"  ✅ Mask prompts: {len(masks)} masks")
        
        print(f"✅ All prompt types tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def main():
    print("🔍 Testing MedSAM2 Ultrasound Inference")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your installation.")
        return False
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed. Please check your checkpoints.")
        return False
    
    # Test inference workflow
    if not test_inference_workflow():
        print("\n❌ Inference test failed. Please check your model setup.")
        return False
    
    print("\n🎉 All tests passed! Your setup is ready for ultrasound inference.")
    print("\nUsage:")
    print("python infer_medsam2_ultrasound.py -i ./data -o ./results --prompt_mode auto")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 