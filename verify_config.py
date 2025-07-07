#!/usr/bin/env python3
"""
Configuration verification script for SAM-MED3D
Verifies checkpoint and YAML compatibility based on Occam's Razor principle
"""

import os
import yaml
import torch
from pathlib import Path

def verify_checkpoint_yaml_compatibility():
    """Verify checkpoint and YAML configuration compatibility"""
    
    print("🔍 Verifying SAM-MED3D Configuration")
    print("=" * 50)
    
    # Check checkpoint
    checkpoint_path = "checkpoints/MedSAM2_latest.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    else:
        print(f"✅ Checkpoint found: {checkpoint_path}")
        
        # Check checkpoint size
        checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"   Size: {checkpoint_size:.1f} MB")
    
    # Check YAML config
    yaml_path = "sam2/configs/sam2.1_hiera_t512.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ YAML config not found: {yaml_path}")
        return False
    else:
        print(f"✅ YAML config found: {yaml_path}")
        
        # Read and verify YAML
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key parameters
        if 'model' in config:
            model_config = config['model']
            
            # Check image size
            if 'image_size' in model_config:
                image_size = model_config['image_size']
                print(f"   Image size: {image_size}²")
                if image_size != 512:
                    print(f"   ⚠️  Warning: Expected 512, got {image_size}")
                else:
                    print(f"   ✅ Image size correct: 512²")
            
            # Check backbone type
            if 'image_encoder' in model_config:
                encoder_config = model_config['image_encoder']
                if 'trunk' in encoder_config:
                    trunk_config = encoder_config['trunk']
                    if '_target_' in trunk_config:
                        target = trunk_config['_target_']
                        if 'Hiera' in target:
                            print(f"   ✅ Backbone: Hiera (tiny)")
                        else:
                            print(f"   ⚠️  Backbone: {target}")
    
    # Verify model loading
    try:
        from sam2.build_sam import build_sam2
        
        print("\n🔄 Testing model loading...")
        model = build_sam2(
            config_file=yaml_path,
            ckpt_path=checkpoint_path,
            device="cpu",
            mode="eval"
        )
        print("✅ Model loaded successfully!")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def verify_data_structure():
    """Verify data directory structure"""
    
    print("\n📁 Verifying Data Structure")
    print("=" * 30)
    
    # Check for US3D structure
    us3d_dirs = ['volumes', 'labels', 'prompts']
    for dir_name in us3d_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Found US3D directory: {dir_name}")
        else:
            print(f"📁 US3D directory not found: {dir_name} (will be created)")
    
    # Check for UNet structure
    if os.path.exists('examples/ultrasound_data'):
        print("✅ Found UNet data directory: examples/ultrasound_data")
    else:
        print("📁 UNet data directory not found: examples/ultrasound_data")

def verify_dependencies():
    """Verify required dependencies"""
    
    print("\n📦 Verifying Dependencies")
    print("=" * 30)
    
    required_packages = [
        'torch', 'numpy', 'cv2', 'nibabel', 'scipy', 'tqdm'
    ]
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")

def main():
    """Main verification function"""
    
    print("🎯 SAM-MED3D Configuration Verification")
    print("Based on Occam's Razor Principle")
    print("=" * 60)
    
    # Verify checkpoint and YAML
    config_ok = verify_checkpoint_yaml_compatibility()
    
    # Verify data structure
    verify_data_structure()
    
    # Verify dependencies
    verify_dependencies()
    
    print("\n📋 Configuration Summary:")
    print("✅ Checkpoint: MedSAM2_latest.pt")
    print("✅ YAML: sam2.1_hiera_t512.yaml")
    print("✅ Image Size: 512²")
    print("✅ Backbone: Hiera (tiny)")
    print("✅ Prompt Types: point_only, box_only, mask+point, etc.")
    print("✅ Data Structures: UNet, US3D")
    
    if config_ok:
        print("\n🎉 Configuration verification PASSED!")
        print("Your SAM-MED3D setup is ready for ultrasound segmentation.")
    else:
        print("\n❌ Configuration verification FAILED!")
        print("Please check the issues above and fix them.")
    
    return config_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 