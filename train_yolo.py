#!/usr/bin/env python3
"""
YOLOv8 Training Script for Malaria Parasite Detection
Clinical-focused training with parasite counting and detection.
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import wandb

# Add src to path
sys.path.append('src')

from models.yolov8_malaria import MalariaYOLOv8
from data.yolo_dataset import MalariaYOLODataset


class MalariaYOLOTrainer:
    """Complete YOLOv8 training pipeline for malaria detection."""
    
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup logging
        self.setup_logging()
        
        # Setup experiment tracking
        self.setup_experiment_tracking()
        
        # Initialize model
        self.setup_model()
        
        # Setup data
        self.setup_data()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'yolo_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_experiment_tracking(self):
        """Setup experiment tracking with wandb."""
        if self.config['logging'].get('use_wandb', False):
            wandb.init(
                project=self.config['logging']['wandb_project'],
                config=self.config,
                name=f"yolov8_malaria_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def setup_model(self):
        """Initialize YOLOv8 model."""
        model_config = self.config.get('model', {})
        
        self.model = MalariaYOLOv8(
            model_size=model_config.get('size', 'n'),
            num_classes=model_config.get('num_classes', 1),
            pretrained=model_config.get('pretrained', True)
        )
        
        self.logger.info(f"YOLOv8{model_config.get('size', 'n')} model initialized")
        
    def setup_data(self):
        """Setup data configuration."""
        data_config = self.config.get('data', {})
        
        # Initialize dataset handler
        self.dataset = MalariaYOLODataset(data_config['data_dir'])
        
        # Create or validate data configuration
        self.data_yaml_path = self.dataset.create_data_yaml()
        
        # Validate dataset
        for split in ['train', 'val']:
            stats = self.dataset.validate_dataset(split)
            self.logger.info(f"{split.capitalize()} dataset stats: {stats}")
            
            if stats['total_images'] == 0:
                self.logger.warning(f"No images found in {split} split!")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting YOLOv8 training...")
        
        # Training parameters
        train_config = self.config['training']
        
        # Clinical-focused training arguments
        train_args = {
            'data': self.data_yaml_path,
            'epochs': train_config['epochs'],
            'imgsz': train_config['image_size'],
            'batch': train_config['batch_size'],
            'lr0': train_config['learning_rate'],
            'weight_decay': train_config['weight_decay'],
            'patience': train_config.get('patience', 20),
            'save_period': train_config.get('save_period', 10),
            'val': True,
            'plots': True,
            'device': self.device,
            'project': train_config['output_dir'],
            'name': f"malaria_yolov8_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            # Medical imaging optimized augmentations
            'hsv_h': 0.01,   # Minimal hue changes for medical accuracy
            'hsv_s': 0.5,    # Moderate saturation changes
            'hsv_v': 0.3,    # Moderate brightness changes
            'degrees': 5.0,  # Small rotations only
            'translate': 0.05, # Minimal translation
            'scale': 0.1,    # Small scale changes
            'shear': 1.0,    # Minimal shear
            'perspective': 0.0, # No perspective changes
            'flipud': 0.0,   # No vertical flips (preserve orientation)
            'fliplr': 0.5,   # Horizontal flips OK
            'mosaic': 0.3,   # Reduced mosaic for medical images
            'mixup': 0.05,   # Minimal mixup
            'copy_paste': 0.0, # No copy-paste for medical accuracy
            
            # Clinical detection thresholds
            'conf': train_config.get('conf_threshold', 0.25),
            'iou': train_config.get('iou_threshold', 0.7),
        }
        
        # Add wandb integration if enabled
        if self.config['logging'].get('use_wandb', False):
            train_args['wandb'] = True
        
        # Start training
        try:
            results = self.model.train(**train_args)
            
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Results saved to: {results.save_dir}")
            
            # Log final metrics
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                self.logger.info(f"Final mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
                self.logger.info(f"Final mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def validate_model(self, model_path: str = None):
        """Validate trained model."""
        self.logger.info("Validating model...")
        
        if model_path:
            # Load specific model
            from ultralytics import YOLO
            model = YOLO(model_path)
        else:
            model = self.model.model
        
        # Run validation
        results = model.val(
            data=self.data_yaml_path,
            device=self.device,
            plots=True
        )
        
        self.logger.info("Validation completed")
        return results
    
    def clinical_evaluation(self, model_path: str, test_images_dir: str):
        """Perform clinical evaluation with parasite counting."""
        self.logger.info("Starting clinical evaluation...")
        
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        # Run predictions on test images
        results = model.predict(
            source=test_images_dir,
            conf=0.25,
            iou=0.7,
            save=True,
            device=self.device
        )
        
        # Analyze results
        clinical_results = []
        
        for result in results:
            # Get image info
            img_path = result.path
            img_name = Path(img_path).name
            
            # Count parasites
            parasite_stats = self.model.count_parasites([result])
            clinical_analysis = self.model.clinical_analysis([result])
            
            clinical_results.append({
                'image': img_name,
                'parasite_count': parasite_stats['count'],
                'avg_confidence': parasite_stats['avg_confidence'],
                'infection_level': clinical_analysis['infection_level'],
                'clinical_significance': clinical_analysis['clinical_significance'],
                'confidence_level': clinical_analysis['confidence_level']
            })
        
        # Save clinical results
        results_df = pd.DataFrame(clinical_results)
        results_path = Path(self.config['training']['output_dir']) / 'clinical_evaluation.csv'
        results_df.to_csv(results_path, index=False)
        
        # Summary statistics
        total_images = len(clinical_results)
        positive_images = sum(1 for r in clinical_results if r['parasite_count'] > 0)
        avg_parasites = np.mean([r['parasite_count'] for r in clinical_results])
        
        self.logger.info(f"Clinical Evaluation Summary:")
        self.logger.info(f"  Total images: {total_images}")
        self.logger.info(f"  Positive images: {positive_images} ({positive_images/total_images*100:.1f}%)")
        self.logger.info(f"  Average parasites per image: {avg_parasites:.2f}")
        self.logger.info(f"  Results saved to: {results_path}")
        
        return clinical_results
    
    def export_model(self, model_path: str, formats: list = ['onnx']):
        """Export trained model for deployment."""
        self.logger.info(f"Exporting model to formats: {formats}")
        
        from ultralytics import YOLO
        model = YOLO(model_path)
        
        exported_paths = []
        for format_type in formats:
            try:
                export_path = model.export(
                    format=format_type,
                    imgsz=self.config['training']['image_size'],
                    optimize=True
                )
                exported_paths.append(export_path)
                self.logger.info(f"Model exported to {format_type}: {export_path}")
            except Exception as e:
                self.logger.error(f"Failed to export to {format_type}: {e}")
        
        return exported_paths


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 malaria detection model')
    parser.add_argument('--config', type=str, default='configs/yolov8_malaria.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation on existing model')
    parser.add_argument('--clinical-eval', type=str, default=None,
                       help='Run clinical evaluation on test images directory')
    parser.add_argument('--export', action='store_true',
                       help='Export model after training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = MalariaYOLOTrainer(config)
    
    if args.validate_only:
        # Only run validation
        if args.resume:
            trainer.validate_model(args.resume)
        else:
            trainer.logger.error("--resume required for validation-only mode")
            return
    
    elif args.clinical_eval:
        # Run clinical evaluation
        if not args.resume:
            trainer.logger.error("--resume required for clinical evaluation")
            return
        trainer.clinical_evaluation(args.resume, args.clinical_eval)
    
    else:
        # Full training pipeline
        results = trainer.train()
        
        # Export model if requested
        if args.export and hasattr(results, 'save_dir'):
            best_model = Path(results.save_dir) / 'weights' / 'best.pt'
            if best_model.exists():
                trainer.export_model(str(best_model), ['onnx', 'torchscript'])
        
        # Run validation on best model
        if hasattr(results, 'save_dir'):
            best_model = Path(results.save_dir) / 'weights' / 'best.pt'
            if best_model.exists():
                trainer.validate_model(str(best_model))
    
    # Close wandb if used
    if config['logging'].get('use_wandb', False):
        wandb.finish()


if __name__ == "__main__":
    main()
