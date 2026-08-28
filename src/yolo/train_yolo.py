"""
YOLO Model Training Module

This module provides functionality to train the YOLO model for food spoilage detection.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOTrainer:
    """
    YOLO model trainer for food spoilage detection.
    
    Attributes:
        config_path (Path): Path to the training configuration YAML file
        project_root (Path): Root directory of the project
        config (Dict): Loaded configuration dictionary
    """
    
    def __init__(self, config_path: str = "configs/yolo_config.yaml"):
        """
        Initialize the YOLO trainer.
        
        Args:
            config_path: Path to the training configuration YAML file
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.config_path = self.project_root / config_path
        
        # Validate config file exists
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}\n"
                f"Please create the config file first."
            )
        
        # Load configuration
        self.config = self._load_config()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized YOLO trainer with config: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load training configuration from YAML file.
        
        Returns:
            Dictionary containing training configuration
        
        Raises:
            ValueError: If config file is empty or invalid
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                raise ValueError("Configuration file is empty")
            
            logger.info("Configuration loaded successfully")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            raise ValueError(f"Failed to parse config file: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _validate_config(self) -> None:
        """
        Validate that all required configuration parameters are present.
        
        Raises:
            ValueError: If required configuration parameters are missing
        """
        required_fields = ["model", "data", "epochs", "imgsz"]
        missing_fields = [field for field in required_fields if field not in self.config]
        
        if missing_fields:
            raise ValueError(
                f"Missing required configuration fields: {', '.join(missing_fields)}\n"
                f"Required fields: {', '.join(required_fields)}"
            )
        
        # Validate data file exists
        data_path = self.project_root / self.config["data"]
        if not data_path.exists():
            raise FileNotFoundError(f"Data config file not found: {data_path}")
        
        logger.info("Configuration validation passed")
    
    def train(
        self,
        resume: bool = False,
        device: Optional[str] = None,
        save_best: bool = True
    ) -> YOLO:
        """
        Train the YOLO model with the specified configuration.
        
        Args:
            resume: Whether to resume training from last checkpoint
            device: Device to use for training (e.g., '0' for GPU, 'cpu')
            save_best: Whether to save the best model
        
        Returns:
            Trained YOLO model instance
        """
        logger.info("Starting YOLO model training...")
        logger.info(f"Configuration: {self.config}")
        
        try:
            # Initialize model
            model = YOLO(self.config["model"])
            logger.info(f"Model initialized: {self.config['model']}")
            
            # Prepare training arguments
            train_args = {
                "data": str(self.project_root / self.config["data"]),
                "epochs": self.config["epochs"],
                "imgsz": self.config["imgsz"],
                "batch": self.config.get("batch", 16),
                "workers": self.config.get("workers", 8),
                "project": str(self.project_root / self.config.get("project", "runs/detect")),
                "name": self.config.get("name", "train"),
                "patience": self.config.get("patience", 50),
                "save": save_best,
                "verbose": True
            }
            
            # Handle resume - if resume=True in config, use last checkpoint
            if self.config.get("resume", False) or resume:
                # Find the latest checkpoint
                project_dir = self.project_root / self.config.get("project", "runs/detect")
                name = self.config.get("name", "train")
                
                # Look for existing runs
                import glob
                existing_runs = sorted(glob.glob(str(project_dir / f"{name}*")))
                
                if existing_runs:
                    latest_run = Path(existing_runs[-1])
                    checkpoint = latest_run / "weights" / "last.pt"
                    
                    if checkpoint.exists():
                        logger.info(f"Resuming from checkpoint: {checkpoint}")
                        train_args["resume"] = str(checkpoint)
                    else:
                        logger.warning(f"No checkpoint found at {checkpoint}, starting fresh")
                        train_args["resume"] = False
                else:
                    train_args["resume"] = False
            else:
                train_args["resume"] = False
            
            # Add device if specified
            if device:
                train_args["device"] = device
            elif "device" in self.config:
                train_args["device"] = self.config["device"]
            
            # Add optional parameters if present in config
            optional_params = [
                "optimizer", "lr0", "lrf", "momentum", "weight_decay",
                "warmup_epochs", "warmup_momentum", "box", "cls", "dfl",
                "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup",
                "copy_paste", "augment", "cache", "rect", "cos_lr", "dropout",
                "plots", "val", "close_mosaic"
            ]
            
            for param in optional_params:
                if param in self.config:
                    train_args[param] = self.config[param]
            
            # Memory optimization: ensure plots are disabled if not explicitly set
            if "plots" not in train_args:
                train_args["plots"] = False
                logger.info("Plots disabled for memory efficiency")
            
            logger.info(f"Training with arguments: {train_args}")
            
            # Train the model
            results = model.train(**train_args)
            
            logger.info("Training completed successfully!")
            
            # Copy best model to models directory
            if save_best:
                self._save_best_model(train_args["project"], train_args["name"])
            
            return model
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _save_best_model(self, project: str, name: str) -> None:
        """
        Copy the best trained model to the models directory.
        
        Args:
            project: Project directory path
            name: Training run name
        """
        try:
            # Path to the best model from training
            best_model_path = Path(project) / name / "weights" / "best.pt"
            
            if not best_model_path.exists():
                logger.warning(f"Best model not found at {best_model_path}")
                return
            
            # Destination path
            models_dir = self.project_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            dest_path = models_dir / "yolo_best.pt"
            
            # Copy the model
            shutil.copy2(best_model_path, dest_path)
            logger.info(f"Best model saved to: {dest_path}")
            
            # Also save last model
            last_model_path = Path(project) / name / "weights" / "last.pt"
            if last_model_path.exists():
                shutil.copy2(last_model_path, models_dir / "yolo_last.pt")
                logger.info(f"Last model saved to: {models_dir / 'yolo_last.pt'}")
                
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")


def train(
    config_path: Optional[str] = None,
    resume: bool = False,
    device: Optional[str] = None
) -> YOLO:
    """
    Convenience function to train YOLO model.
    
    Args:
        config_path: Path to config file (default: configs/yolo_config.yaml)
        resume: Whether to resume training from last checkpoint
        device: Device to use for training
    
    Returns:
        Trained YOLO model instance
    """
    kwargs = {}
    if config_path:
        kwargs['config_path'] = config_path
    
    trainer = YOLOTrainer(**kwargs)
    return trainer.train(resume=resume, device=device)


if __name__ == "__main__":
    # Train the model
    logger.info("=" * 60)
    logger.info("YOLO Model Training - Food Spoilage Detection")
    logger.info("=" * 60)
    
    try:
        model = train()
        logger.info("\n" + "=" * 60)
        logger.info("Training completed! Model saved to models/yolo_best.pt")
        logger.info("Run evaluation with: python src/yolo/evaluate_yolo.py")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        raise
