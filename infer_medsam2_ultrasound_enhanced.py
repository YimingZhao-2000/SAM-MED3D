#!/usr/bin/env python3
"""
Enhanced MedSAM2 Ultrasound Inference with Loss Calculation and Monitoring
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import cv2

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultrasound_data_loader import UltrasoundDataLoader
from sam2.sam2_image_predictor import SAM2ImagePredictor

class EnhancedMedSAM2Inference:
    """Enhanced MedSAM2 inference with loss calculation and monitoring"""
    
    def __init__(self, 
                 checkpoint_path: str = "checkpoints/MedSAM2_latest.pt",
                 device: str = "cuda:0",
                 config_path: str = "sam2/configs/sam2.1_hiera_tiny_finetune512.yaml"):
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        # Initialize predictor
        self.predictor = SAM2ImagePredictor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=self.device
        )
        
        # Metrics tracking
        self.metrics = {
            'dice_scores': [],
            'iou_scores': [],
            'hausdorff_distances': [],
            'processing_times': [],
            'file_names': []
        }
        
        # Setup logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('MedSAM2Inference')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def calculate_dice_score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate Dice score between prediction and ground truth"""
        intersection = np.logical_and(pred, gt).sum()
        union = pred.sum() + gt.sum()
        return (2.0 * intersection) / (union + 1e-8)
    
    def calculate_iou_score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate IoU score between prediction and ground truth"""
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return intersection / (union + 1e-8)
    
    def calculate_hausdorff_distance(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate Hausdorff distance between prediction and ground truth"""
        try:
            from scipy.spatial.distance import directed_hausdorff
            
            # Get coordinates of foreground pixels
            pred_coords = np.argwhere(pred > 0)
            gt_coords = np.argwhere(gt > 0)
            
            if len(pred_coords) == 0 or len(gt_coords) == 0:
                return float('inf')
            
            # Calculate directed Hausdorff distances
            d1, _, _ = directed_hausdorff(pred_coords, gt_coords)
            d2, _, _ = directed_hausdorff(gt_coords, pred_coords)
            
            return max(d1, d2)
        except ImportError:
            self.logger.warning("scipy not available, skipping Hausdorff distance")
            return 0.0
    
    def process_single_volume(self, 
                            volume_data: np.ndarray, 
                            label_data: Optional[np.ndarray] = None,
                            prompts: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Process a single 3D volume slice by slice"""
        
        start_time = time.time()
        height, width, depth = volume_data.shape
        
        # Initialize output volume
        segmentation_volume = np.zeros((height, width, depth), dtype=np.uint8)
        
        # Process each slice
        for slice_idx in range(depth):
            # Extract 2D slice
            slice_data = volume_data[:, :, slice_idx]
            
            # Normalize slice to [0, 255] for SAM2
            slice_normalized = self._normalize_slice(slice_data)
            
            # Prepare prompts for this slice
            slice_prompts = self._prepare_slice_prompts(prompts, slice_idx) if prompts else None
            
            # Run SAM2 inference
            try:
                masks, scores, logits = self.predictor.predict(
                    point_coords=slice_prompts.get('points', None) if slice_prompts else None,
                    point_labels=slice_prompts.get('point_labels', None) if slice_prompts else None,
                    box=slice_prompts.get('box', None) if slice_prompts else None,
                    mask_input=slice_prompts.get('mask_input', None) if slice_prompts else None,
                    multimask_output=False,
                    return_logits=True
                )
                
                # Use the best mask
                if len(masks) > 0:
                    best_mask = masks[np.argmax(scores)]
                    segmentation_volume[:, :, slice_idx] = best_mask.astype(np.uint8)
                    
            except Exception as e:
                self.logger.warning(f"Error processing slice {slice_idx}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # Calculate metrics if ground truth is available
        metrics = {
            'processing_time': processing_time,
            'volume_shape': volume_data.shape,
            'segmentation_shape': segmentation_volume.shape
        }
        
        if label_data is not None:
            metrics.update({
                'dice_score': self.calculate_dice_score(segmentation_volume, label_data),
                'iou_score': self.calculate_iou_score(segmentation_volume, label_data),
                'hausdorff_distance': self.calculate_hausdorff_distance(segmentation_volume, label_data)
            })
        
        return segmentation_volume, metrics
    
    def _normalize_slice(self, slice_data: np.ndarray) -> np.ndarray:
        """Normalize slice data to [0, 255] range"""
        slice_min = slice_data.min()
        slice_max = slice_data.max()
        
        if slice_max > slice_min:
            normalized = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(slice_data, dtype=np.uint8)
        
        return normalized
    
    def _prepare_slice_prompts(self, prompts: Dict, slice_idx: int) -> Dict:
        """Prepare prompts for a specific slice"""
        slice_prompts = {}
        
        if 'points' in prompts:
            # Extract points for this slice
            points_3d = np.array(prompts['points'])
            slice_points = points_3d[points_3d[:, 2] == slice_idx]
            if len(slice_points) > 0:
                slice_prompts['points'] = slice_points[:, :2]  # Only x, y coordinates
                slice_prompts['point_labels'] = [1] * len(slice_points)
        
        if 'box' in prompts:
            # Use the same box for all slices (2D box)
            slice_prompts['box'] = prompts['box']
        
        if 'mask_input' in prompts:
            # Extract mask for this slice
            mask_3d = prompts['mask_input']
            if mask_3d.shape[2] > slice_idx:
                slice_prompts['mask_input'] = mask_3d[:, :, slice_idx]
        
        return slice_prompts
    
    def run_inference(self, 
                     input_dir: str,
                     output_dir: str,
                     calculate_loss: bool = True,
                     save_metrics: bool = True,
                     log_file: Optional[str] = None,
                     metrics_file: Optional[str] = None) -> Dict:
        """Run inference on all volumes in the input directory"""
        
        # Setup logging to file if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(file_handler)
        
        # Load data
        data_loader = UltrasoundDataLoader(input_dir)
        
        if not data_loader.data_files:
            self.logger.error("No data files found!")
            return {}
        
        self.logger.info(f"Found {len(data_loader.data_files)} data-label pairs")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each file
        for idx, (data_file, label_file) in enumerate(tqdm(data_loader.data_files, desc="Processing volumes")):
            try:
                # Load data
                volume_data, label_data = data_loader.load_volume_and_label(data_file, label_file)
                
                if volume_data is None:
                    self.logger.warning(f"Failed to load volume: {data_file}")
                    continue
                
                # Generate prompts from ground truth if available
                prompts = None
                if calculate_loss and label_data is not None:
                    prompts = self._generate_prompts_from_gt(label_data)
                
                # Process volume
                segmentation, metrics = self.process_single_volume(
                    volume_data, label_data, prompts
                )
                
                # Save results
                output_file = Path(output_dir) / f"{Path(data_file).stem}_seg.nii.gz"
                self._save_segmentation(segmentation, output_file)
                
                # Store metrics
                file_name = Path(data_file).name
                self.metrics['file_names'].append(file_name)
                self.metrics['processing_times'].append(metrics['processing_time'])
                
                if calculate_loss and label_data is not None:
                    self.metrics['dice_scores'].append(metrics.get('dice_score', 0))
                    self.metrics['iou_scores'].append(metrics.get('iou_score', 0))
                    self.metrics['hausdorff_distances'].append(metrics.get('hausdorff_distance', 0))
                
                self.logger.info(f"Processed {file_name}: Dice={metrics.get('dice_score', 'N/A'):.4f}, "
                               f"Time={metrics['processing_time']:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error processing {data_file}: {e}")
                continue
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics()
        
        # Save metrics if requested
        if save_metrics and metrics_file:
            self._save_metrics(overall_metrics, metrics_file)
        
        return overall_metrics
    
    def _generate_prompts_from_gt(self, label_data: np.ndarray) -> Dict:
        """Generate prompts from ground truth labels"""
        prompts = {}
        
        # Find center of mass for each slice
        points_3d = []
        for slice_idx in range(label_data.shape[2]):
            slice_label = label_data[:, :, slice_idx]
            if slice_label.sum() > 0:
                # Calculate center of mass
                y_coords, x_coords = np.where(slice_label > 0)
                if len(y_coords) > 0 and len(x_coords) > 0:
                    center_y = int(np.mean(y_coords))
                    center_x = int(np.mean(x_coords))
                    points_3d.append([center_x, center_y, slice_idx])
        
        if points_3d:
            prompts['points'] = points_3d
            prompts['point_labels'] = [1] * len(points_3d)
        
        return prompts
    
    def _save_segmentation(self, segmentation: np.ndarray, output_file: Path):
        """Save segmentation result"""
        try:
            import nibabel as nib
            nii_img = nib.Nifti1Image(segmentation, affine=np.eye(4))
            nib.save(nii_img, output_file)
        except ImportError:
            # Fallback to numpy save
            np.save(output_file.with_suffix('.npy'), segmentation)
    
    def _calculate_overall_metrics(self) -> Dict:
        """Calculate overall metrics across all processed files"""
        metrics = {
            'total_files': len(self.metrics['file_names']),
            'total_processing_time': sum(self.metrics['processing_times']),
            'average_processing_time': np.mean(self.metrics['processing_times'])
        }
        
        if self.metrics['dice_scores']:
            metrics.update({
                'dice_score': np.mean(self.metrics['dice_scores']),
                'dice_score_std': np.std(self.metrics['dice_scores']),
                'iou_score': np.mean(self.metrics['iou_scores']),
                'iou_score_std': np.std(self.metrics['iou_scores']),
                'hausdorff_distance': np.mean(self.metrics['hausdorff_distances']),
                'hausdorff_distance_std': np.std(self.metrics['hausdorff_distances'])
            })
        
        return metrics
    
    def _save_metrics(self, metrics: Dict, metrics_file: str):
        """Save metrics to JSON file"""
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot metrics if matplotlib is available"""
        try:
            if not self.metrics['dice_scores']:
                self.logger.warning("No metrics available for plotting")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Dice scores
            axes[0, 0].hist(self.metrics['dice_scores'], bins=20, alpha=0.7)
            axes[0, 0].set_title('Dice Score Distribution')
            axes[0, 0].set_xlabel('Dice Score')
            axes[0, 0].set_ylabel('Frequency')
            
            # IoU scores
            axes[0, 1].hist(self.metrics['iou_scores'], bins=20, alpha=0.7)
            axes[0, 1].set_title('IoU Score Distribution')
            axes[0, 1].set_xlabel('IoU Score')
            axes[0, 1].set_ylabel('Frequency')
            
            # Processing times
            axes[1, 0].hist(self.metrics['processing_times'], bins=20, alpha=0.7)
            axes[1, 0].set_title('Processing Time Distribution')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            
            # Hausdorff distances
            axes[1, 1].hist(self.metrics['hausdorff_distances'], bins=20, alpha=0.7)
            axes[1, 1].set_title('Hausdorff Distance Distribution')
            axes[1, 1].set_xlabel('Distance')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Metrics plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plot generation")

def main():
    parser = argparse.ArgumentParser(description="Enhanced MedSAM2 Ultrasound Inference")
    parser.add_argument("-i", "--input", required=True, help="Input directory with .nrrd files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for results")
    parser.add_argument("--checkpoint", default="checkpoints/MedSAM2_latest.pt", help="Model checkpoint path")
    parser.add_argument("--device", default="cuda:0", help="Device to use (cuda:0, cpu)")
    parser.add_argument("--calculate_loss", action="store_true", help="Calculate loss metrics")
    parser.add_argument("--save_metrics", action="store_true", help="Save metrics to file")
    parser.add_argument("--log_file", help="Log file path")
    parser.add_argument("--metrics_file", help="Metrics file path")
    parser.add_argument("--plot_metrics", help="Save metrics plot to file")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = EnhancedMedSAM2Inference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Run inference
    metrics = inference.run_inference(
        input_dir=args.input,
        output_dir=args.output,
        calculate_loss=args.calculate_loss,
        save_metrics=args.save_metrics,
        log_file=args.log_file,
        metrics_file=args.metrics_file
    )
    
    # Print summary
    print("\n" + "="*50)
    print("📊 INFERENCE SUMMARY")
    print("="*50)
    print(f"Total files processed: {metrics.get('total_files', 0)}")
    print(f"Total processing time: {metrics.get('total_processing_time', 0):.2f}s")
    print(f"Average processing time: {metrics.get('average_processing_time', 0):.2f}s")
    
    if 'dice_score' in metrics:
        print(f"Average Dice Score: {metrics['dice_score']:.4f} ± {metrics['dice_score_std']:.4f}")
        print(f"Average IoU Score: {metrics['iou_score']:.4f} ± {metrics['iou_score_std']:.4f}")
        print(f"Average Hausdorff Distance: {metrics['hausdorff_distance']:.4f} ± {metrics['hausdorff_distance_std']:.4f}")
    
    # Generate plot if requested
    if args.plot_metrics:
        inference.plot_metrics(args.plot_metrics)
    
    print("="*50)
    print("✅ Inference completed successfully!")

if __name__ == "__main__":
    main() 