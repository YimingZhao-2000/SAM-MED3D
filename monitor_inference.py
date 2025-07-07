#!/usr/bin/env python3
"""
Monitoring script for MedSAM2 ultrasound inference
Tracks progress and calculates metrics after inference
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

def calculate_metrics(pred_file: Path, gt_file: Path) -> dict:
    """Calculate metrics between prediction and ground truth"""
    try:
        import nibabel as nib
        
        # Load prediction and ground truth
        pred_nib = nib.load(pred_file)
        gt_nib = nib.load(gt_file)
        
        pred_data = pred_nib.get_fdata().astype(bool)
        gt_data = gt_nib.get_fdata().astype(bool)
        
        # Calculate metrics
        intersection = np.logical_and(pred_data, gt_data).sum()
        union = np.logical_or(pred_data, gt_data).sum()
        
        dice_score = (2.0 * intersection) / (pred_data.sum() + gt_data.sum() + 1e-8)
        iou_score = intersection / (union + 1e-8)
        
        return {
            'dice_score': float(dice_score),
            'iou_score': float(iou_score),
            'prediction_voxels': int(pred_data.sum()),
            'ground_truth_voxels': int(gt_data.sum()),
            'intersection_voxels': int(intersection)
        }
        
    except Exception as e:
        print(f"Error calculating metrics for {pred_file}: {e}")
        return {}

def monitor_inference(input_dir: str, output_dir: str, results_file: str = "inference_metrics.json"):
    """Monitor inference progress and calculate metrics"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    if not output_path.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    # Find all input files
    input_files = list(input_path.glob("*.nrrd"))
    input_files = [f for f in input_files if not f.name.endswith("_Mask.seg.nrrd")]
    
    print(f"📊 Found {len(input_files)} input files")
    
    # Track metrics
    metrics = {
        'total_files': len(input_files),
        'processed_files': 0,
        'successful_files': 0,
        'failed_files': 0,
        'file_metrics': [],
        'overall_metrics': {}
    }
    
    # Check each input file
    for input_file in tqdm(input_files, desc="Checking inference results"):
        base_name = input_file.stem
        pred_file = output_path / f"{base_name}_seg.nii.gz"
        
        # Check if prediction exists
        if pred_file.exists():
            metrics['processed_files'] += 1
            
            # Look for corresponding ground truth
            gt_file = input_path / f"{base_name}_Mask.seg.nrrd"
            
            if gt_file.exists():
                # Calculate metrics
                file_metrics = calculate_metrics(pred_file, gt_file)
                if file_metrics:
                    file_metrics['file_name'] = input_file.name
                    file_metrics['prediction_file'] = str(pred_file)
                    file_metrics['ground_truth_file'] = str(gt_file)
                    metrics['file_metrics'].append(file_metrics)
                    metrics['successful_files'] += 1
                else:
                    metrics['failed_files'] += 1
            else:
                print(f"⚠️  No ground truth found for {input_file.name}")
                metrics['successful_files'] += 1  # Still successful if prediction exists
        else:
            print(f"❌ No prediction found for {input_file.name}")
            metrics['failed_files'] += 1
    
    # Calculate overall metrics
    if metrics['file_metrics']:
        dice_scores = [m['dice_score'] for m in metrics['file_metrics']]
        iou_scores = [m['iou_score'] for m in metrics['file_metrics']]
        
        metrics['overall_metrics'] = {
            'mean_dice_score': float(np.mean(dice_scores)),
            'std_dice_score': float(np.std(dice_scores)),
            'mean_iou_score': float(np.mean(iou_scores)),
            'std_iou_score': float(np.std(iou_scores)),
            'min_dice_score': float(np.min(dice_scores)),
            'max_dice_score': float(np.max(dice_scores)),
            'min_iou_score': float(np.min(iou_scores)),
            'max_iou_score': float(np.max(iou_scores))
        }
    
    # Save metrics
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 INFERENCE MONITORING SUMMARY")
    print("="*60)
    print(f"Total input files: {metrics['total_files']}")
    print(f"Processed files: {metrics['processed_files']}")
    print(f"Successful files: {metrics['successful_files']}")
    print(f"Failed files: {metrics['failed_files']}")
    
    if metrics['overall_metrics']:
        print(f"\n📈 OVERALL METRICS")
        print(f"Dice Score: {metrics['overall_metrics']['mean_dice_score']:.4f} ± {metrics['overall_metrics']['std_dice_score']:.4f}")
        print(f"IoU Score: {metrics['overall_metrics']['mean_iou_score']:.4f} ± {metrics['overall_metrics']['std_iou_score']:.4f}")
        print(f"Dice Score Range: [{metrics['overall_metrics']['min_dice_score']:.4f}, {metrics['overall_metrics']['max_dice_score']:.4f}]")
        print(f"IoU Score Range: [{metrics['overall_metrics']['min_iou_score']:.4f}, {metrics['overall_metrics']['max_iou_score']:.4f}]")
    
    print(f"\n📁 Results saved to: {results_file}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Monitor MedSAM2 inference progress")
    parser.add_argument("-i", "--input", required=True, help="Input directory with .nrrd files")
    parser.add_argument("-o", "--output", required=True, help="Output directory with results")
    parser.add_argument("-r", "--results", default="inference_metrics.json", help="Results file path")
    
    args = parser.parse_args()
    
    monitor_inference(args.input, args.output, args.results)

if __name__ == "__main__":
    main() 