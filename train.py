#!/usr/bin/env python3
"""
Malaria Detection Training Script
Production-ready training with clinical focus on high sensitivity.
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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import wandb

# Add src to path
sys.path.append('src')

from data.dataset import create_data_loaders, analyze_dataset_balance
from models.efficientnet import create_model, FocalLoss, ClinicalLoss
from utils.metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve
from utils.gradcam import GradCAM, visualize_gradcam


class MalariaTrainer:
    """Complete training pipeline for malaria detection."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Setup experiment tracking
        self.setup_experiment_tracking()
        
        # Initialize model, data, optimizer
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.best_auc = 0.0
        self.best_sensitivity = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_experiment_tracking(self):
        """Setup experiment tracking with wandb and tensorboard."""
        # Tensorboard
        self.tb_writer = SummaryWriter(
            log_dir=f"{self.config['logging']['tensorboard_dir']}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Wandb (optional)
        if self.config['logging'].get('use_wandb', False):
            wandb.init(
                project=self.config['logging']['wandb_project'],
                config=self.config,
                name=f"malaria_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def setup_model(self):
        """Initialize model and loss function."""
        self.model = create_model(
            model_name=self.config['model']['name'],
            num_classes=self.config['model']['num_classes'],
            pretrained=self.config['model']['pretrained'],
            dropout=self.config['model']['dropout'],
            use_timm=self.config['model'].get('use_timm', False)
        ).to(self.device)
        
        # Loss function
        loss_type = self.config['training']['loss_function']
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config['training']['focal_alpha'],
                gamma=self.config['training']['focal_gamma']
            )
        elif loss_type == 'clinical':
            self.criterion = ClinicalLoss(
                sensitivity_weight=self.config['training']['sensitivity_weight'],
                specificity_weight=self.config['training']['specificity_weight']
            )
        
        self.logger.info(f"Model: {self.config['model']['name']}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_data(self):
        """Setup data loaders."""
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_dir=self.config['data']['data_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            image_size=self.config['data']['image_size']
        )
        
        # Analyze dataset balance
        balance = analyze_dataset_balance(self.config['data']['data_dir'])
        self.logger.info("Dataset balance:")
        for split, stats in balance.items():
            self.logger.info(f"  {split}: {stats['parasitized']} parasitized, "
                           f"{stats['uninfected']} uninfected (ratio: {stats['balance_ratio']:.3f})")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Optimizer
        if self.config['training']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        
        # Scheduler
        scheduler_type = self.config['training']['scheduler']
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=self.config['training']['scheduler_patience'],
                verbose=True
            )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch progress
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.4f}'
                )
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), labels)
                
                total_loss += loss.item()
                
                # Collect predictions for metrics
                probs = torch.sigmoid(outputs.squeeze())
                all_outputs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Calculate metrics
        metrics = calculate_metrics(np.array(all_labels), np.array(all_outputs))
        self.val_aucs.append(metrics['auc'])
        
        return avg_loss, metrics
    
    def save_checkpoint(self, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_auc': self.best_auc,
            'best_sensitivity': self.best_sensitivity,
            'config': self.config,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')
            self.logger.info(f"New best model saved with AUC: {metrics['auc']:.4f}")
    
    def log_metrics(self, train_loss, val_loss, metrics):
        """Log metrics to tensorboard and wandb."""
        # Tensorboard
        self.tb_writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
        self.tb_writer.add_scalar('Loss/Validation', val_loss, self.current_epoch)
        self.tb_writer.add_scalar('Metrics/AUC', metrics['auc'], self.current_epoch)
        self.tb_writer.add_scalar('Metrics/Accuracy', metrics['accuracy'], self.current_epoch)
        self.tb_writer.add_scalar('Metrics/Sensitivity', metrics['sensitivity'], self.current_epoch)
        self.tb_writer.add_scalar('Metrics/Specificity', metrics['specificity'], self.current_epoch)
        self.tb_writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        
        # Wandb
        if self.config['logging'].get('use_wandb', False):
            wandb.log({
                'epoch': self.current_epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'auc': metrics['auc'],
                'accuracy': metrics['accuracy'],
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train and validate
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate_epoch()
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(metrics['auc'])
            else:
                self.scheduler.step()
            
            # Check for best model
            is_best = False
            if metrics['auc'] > self.best_auc:
                self.best_auc = metrics['auc']
                is_best = True
            
            if metrics['sensitivity'] > self.best_sensitivity:
                self.best_sensitivity = metrics['sensitivity']
            
            # Save checkpoint
            self.save_checkpoint(metrics, is_best)
            
            # Log metrics
            self.log_metrics(train_loss, val_loss, metrics)
            
            # Print epoch summary
            self.logger.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'AUC: {metrics["auc"]:.4f}, Sensitivity: {metrics["sensitivity"]:.4f}, '
                f'Specificity: {metrics["specificity"]:.4f}'
            )
            
            # Early stopping
            if self.config['training'].get('early_stopping', False):
                patience = self.config['training']['early_stopping_patience']
                if len(self.val_aucs) > patience:
                    recent_aucs = self.val_aucs[-patience:]
                    if all(auc <= self.best_auc for auc in recent_aucs):
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        self.logger.info(f"Training completed. Best AUC: {self.best_auc:.4f}")
        
        # Close writers
        self.tb_writer.close()
        if self.config['logging'].get('use_wandb', False):
            wandb.finish()


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train malaria detection model')
    parser.add_argument('--config', type=str, default='configs/efficientnet_b0.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = MalariaTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_auc = checkpoint['best_auc']
        trainer.logger.info(f"Resumed from checkpoint at epoch {trainer.current_epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
