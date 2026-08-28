"""
ResNet Training Module for Food Spoilage Classification

This module provides comprehensive training functionality for ResNet models,
including data loading, augmentation, training loop, validation, checkpointing,
and metrics tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import os
import shutil
import sys
import time
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from resnet.model import build_resnet, build_resnet_from_config
from pipeline.utils import (
    load_config, save_config, save_json, setup_logger, 
    get_device, ensure_dir, set_seed, format_time
)


class ResNetTrainer:
    """
    Trainer class for ResNet models with full training pipeline.
    
    Args:
        config_path (str or Path): Path to configuration file
        resume_from (str or Path, optional): Checkpoint to resume from
    
    Example:
        >>> trainer = ResNetTrainer('configs/resnet_config.yaml')
        >>> trainer.train()
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict] = None,
        resume_from: Optional[str] = None
    ):
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        # Setup seed for reproducibility
        seed = self.config.get('seed', 42)
        set_seed(seed)
        
        # Setup device
        device_name = self.config.get('device', 'cuda')
        self.device = get_device(device_name, verbose=True)
        
        # Setup logging
        self._setup_logging()
        self.logger.info("Starting ResNet training pipeline")
        self.logger.info(f"Configuration loaded from: {config_path}")
        
        # Setup directories
        self._setup_directories()
        
        # Build model
        self.model = self._build_model()
        self.logger.info(f"Model architecture: {self.config['model']['architecture']}")
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup loss, optimizer, scheduler
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision scaler
        amp_enabled = self.config.get('mixed_precision', {}).get('enabled', False)
        self.use_amp = amp_enabled and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            self.logger.info("Mixed precision (AMP) enabled")
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('output', {})
        log_dir = Path(log_config.get('project', 'runs/resnet')) / log_config.get('name', 'resnet_train')
        log_file = log_dir / 'training.log'
        
        self.logger = setup_logger(
            name='resnet_trainer',
            log_file=log_file,
            level='INFO',
            console_output=log_config.get('verbose', True)
        )
    
    def _setup_directories(self):
        """Setup output directories."""
        checkpoint_dir = self.config.get('checkpoint', {}).get('save_dir', 'models')
        output_config = self.config.get('output', {})
        
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.output_dir = ensure_dir(Path(output_config.get('project', 'runs/resnet')) / 
                                     output_config.get('name', 'resnet_train'))
        
        # Save config to output directory
        save_config(self.config, self.output_dir / 'config.yaml')
    
    def _build_model(self) -> nn.Module:
        """Build model from configuration."""
        model = build_resnet_from_config(self.config)
        model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _setup_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Setup training and validation data loaders with augmentation."""
        data_config = self.config.get('data', {})
        train_config = self.config.get('training', {})
        aug_config = self.config.get('augmentation', {})
        
        # Training transforms with augmentation
        train_transforms = self._create_train_transforms(aug_config, data_config)
        
        # Validation transforms (no augmentation)
        val_transforms = self._create_val_transforms(aug_config, data_config)
        
        # Load datasets
        train_dir = data_config.get('train_dir', 'data/resnet_dataset/train')
        val_dir = data_config.get('val_dir', 'data/resnet_dataset/val')
        
        if not Path(train_dir).exists():
            raise FileNotFoundError(f"Training data not found: {train_dir}")
        
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Classes: {train_dataset.classes}")
        
        # Class imbalance: compute inverse-frequency weights for WeightedRandomSampler
        class_counts = [0] * len(train_dataset.classes)
        for _, label in train_dataset.samples:
            class_counts[label] += 1
        class_weights = [1.0 / count for count in class_counts]
        sample_weights = [class_weights[label] for _, label in train_dataset.samples]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        self.logger.info(f"Class counts: { {cls: cnt for cls, cnt in zip(train_dataset.classes, class_counts)} }")
        
        # Validation dataset
        val_loader = None
        if self.config.get('validation', {}).get('enabled', True):
            if Path(val_dir).exists():
                val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
                self.logger.info(f"Validation samples: {len(val_dataset)}")
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=train_config.get('batch_size', 32),
                    shuffle=False,
                    num_workers=data_config.get('num_workers', 4),
                    pin_memory=True,
                    persistent_workers=True
                )
            else:
                self.logger.warning(f"Validation directory not found: {val_dir}")
        
        # Training loader (sampler provides balanced sampling; shuffle is incompatible with sampler)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.get('batch_size', 32),
            sampler=sampler,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )
        
        return train_loader, val_loader
    
    def _create_train_transforms(self, aug_config: Dict, data_config: Dict) -> transforms.Compose:
        """Create training data transforms with augmentation."""
        image_size = data_config.get('image_size', 224)
        transform_list = []
        
        # Resize to slightly larger than target so RandomCrop samples different spatial regions
        transform_list.append(transforms.Resize((image_size + 32, image_size + 32)))
        
        # Random crop to final size (genuine spatial diversity, no zero-padding needed)
        if aug_config.get('random_crop', True):
            transform_list.append(transforms.RandomCrop(image_size))
        
        # Random horizontal flip
        if aug_config.get('random_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # Random rotation
        rotation = aug_config.get('random_rotation', 15)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))
        
        # Color jitter
        color_jitter = aug_config.get('color_jitter', {})
        if color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=color_jitter.get('brightness', 0.2),
                contrast=color_jitter.get('contrast', 0.2),
                saturation=color_jitter.get('saturation', 0.2),
                hue=color_jitter.get('hue', 0.1)
            ))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        normalize_config = aug_config.get('normalize', {})
        mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
        std = normalize_config.get('std', [0.229, 0.224, 0.225])
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    def _create_val_transforms(self, aug_config: Dict, data_config: Dict) -> transforms.Compose:
        """Create validation data transforms (no augmentation)."""
        image_size = data_config.get('image_size', 224)
        normalize_config = aug_config.get('normalize', {})
        mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
        std = normalize_config.get('std', [0.229, 0.224, 0.225])
        
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'CrossEntropyLoss')
        
        if loss_type == 'CrossEntropyLoss':
            label_smoothing = loss_config.get('label_smoothing', 0.0)
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_config = self.config.get('optimizer', {})
        train_config = self.config.get('training', {})
        
        optimizer_type = optimizer_config.get('type', 'Adam')
        lr = train_config.get('learning_rate', 0.001)
        weight_decay = train_config.get('weight_decay', 0.0001)
        
        if optimizer_type == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', [0.9, 0.999]),
                eps=optimizer_config.get('eps', 1e-8)
            )
        elif optimizer_type == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=train_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.logger.warning(f"Unknown optimizer: {optimizer_type}, using Adam")
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'StepLR')
        
        if scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config.get('patience', 5),
                factor=scheduler_config.get('gamma', 0.1),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            train_config = self.config.get('training', {})
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config.get('epochs', 50),
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            return None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for images, labels in progress_bar:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                if self.config.get('gradient_clipping', {}).get('enabled', False):
                    max_norm = self.config['gradient_clipping'].get('max_norm', 1.0)
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.config.get('gradient_clipping', {}).get('enabled', False):
                    max_norm = self.config['gradient_clipping'].get('max_norm', 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = total_loss / total
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_loss = total_loss / total
        val_accuracy = 100. * correct / total
        
        return val_loss, val_accuracy
    
    def save_checkpoint(self, is_best: bool = False, filename: str = 'checkpoint.pt'):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Save checkpoint atomically (temp file → rename) to avoid file-lock errors
        checkpoint_path = self.checkpoint_dir / filename
        tmp_checkpoint = checkpoint_path.with_suffix('.tmp')
        torch.save(checkpoint, tmp_checkpoint)
        shutil.move(str(tmp_checkpoint), str(checkpoint_path))
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model atomically
        if is_best:
            best_path = self.checkpoint_dir / 'resnet_spoilage.pt'
            tmp_best = best_path.with_suffix('.tmp')
            torch.save(self.model.state_dict(), tmp_best)
            shutil.move(str(tmp_best), str(best_path))
            self.logger.info(f"Best model saved: {best_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        self.logger.info(f"Starting from epoch {self.current_epoch + 1}")
    
    def _check_early_stopping(self, val_accuracy: float) -> bool:
        """Check if early stopping criteria is met (monitors val_accuracy, higher is better)."""
        early_stop_config = self.config.get('early_stopping', {})
        
        if not early_stop_config.get('enabled', True):
            return False
        
        patience = early_stop_config.get('patience', 10)
        min_delta = early_stop_config.get('min_delta', 0.001)
        
        if val_accuracy > (self.best_val_accuracy + min_delta):
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        
        if self.early_stop_counter >= patience:
            self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            return True
        
        return False
    
    def train(self):
        """Main training loop."""
        train_config = self.config.get('training', {})
        num_epochs = train_config.get('epochs', 50)
        checkpoint_config = self.config.get('checkpoint', {})
        save_frequency = checkpoint_config.get('save_frequency', 5)
        
        self.logger.info("="*60)
        self.logger.info("Starting training")
        self.logger.info(f"Total epochs: {num_epochs}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Training
                train_loss, train_acc = self.train_epoch()
                
                # Validation
                val_loss, val_acc = 0.0, 0.0
                if self.val_loader and self.config.get('validation', {}).get('enabled', True):
                    val_freq = self.config.get('validation', {}).get('frequency', 1)
                    if (epoch + 1) % val_freq == 0:
                        val_loss, val_acc = self.validate()
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss if val_loss > 0 else train_loss)
                    else:
                        self.scheduler.step()
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_accuracy'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                self.history['learning_rate'].append(current_lr)
                
                # Logging
                epoch_time = time.time() - epoch_start
                self.logger.info(
                    f"Epoch [{epoch + 1}/{num_epochs}] | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                    f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
                )
                
                # Save best model
                is_best = False
                monitor_metric = checkpoint_config.get('monitor', 'val_accuracy')
                
                if monitor_metric == 'val_accuracy' and val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    is_best = True
                elif monitor_metric == 'val_loss' and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best = True
                
                # Save checkpoint
                if checkpoint_config.get('save_best_only', True):
                    if is_best:
                        self.save_checkpoint(is_best=True, filename=f'checkpoint_epoch_{epoch + 1}.pt')
                else:
                    if (epoch + 1) % save_frequency == 0:
                        self.save_checkpoint(is_best=is_best, filename=f'checkpoint_epoch_{epoch + 1}.pt')
                
                # Early stopping
                if self.val_loader and self._check_early_stopping(val_acc):
                    break
            
            # Training completed
            total_time = time.time() - start_time
            self.logger.info("="*60)
            self.logger.info("Training completed!")
            self.logger.info(f"Total training time: {format_time(total_time)}")
            self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
            self.logger.info("="*60)
            
            # Save final metrics
            self._save_metrics()
            
        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user")
            self.save_checkpoint(filename='checkpoint_interrupted.pt')
            self._save_metrics()
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_metrics(self):
        """Save training metrics to file."""
        metrics_file = self.output_dir / 'training_metrics.json'
        metrics = {
            'final_epoch': self.current_epoch + 1,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        save_json(metrics, metrics_file)
        self.logger.info(f"Metrics saved to: {metrics_file}")


# ============================================================================
# Standalone Training Function
# ============================================================================

def train_resnet(
    config_path: str = 'configs/resnet_config.yaml',
    resume_from: Optional[str] = None
):
    """
    Train ResNet model using configuration file.
    
    Args:
        config_path (str): Path to configuration file
        resume_from (str, optional): Checkpoint path to resume from
    
    Example:
        >>> train_resnet('configs/resnet_config.yaml')
        >>> train_resnet('configs/resnet_config.yaml', resume_from='models/checkpoint.pt')
    """
    trainer = ResNetTrainer(config_path=config_path, resume_from=resume_from)
    trainer.train()


# ============================================================================
# Main / CLI
# ============================================================================

def main():
    """Main function for CLI training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ResNet for Food Spoilage Detection')
    parser.add_argument('--config', type=str, default='configs/resnet_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, overrides config)')
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Override device if specified
        if args.device:
            config['device'] = args.device
        
        # Train
        trainer = ResNetTrainer(config_dict=config, resume_from=args.resume)
        trainer.train()
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
