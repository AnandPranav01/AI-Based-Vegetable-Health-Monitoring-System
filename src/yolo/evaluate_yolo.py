"""
YOLO Model Evaluation Module

This module provides functionality to evaluate the YOLO model for food spoilage detection.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOEvaluator:
    """
    YOLO model evaluator for food spoilage detection.
    
    Attributes:
        model_path (Path): Path to the trained YOLO model
        data_config (Path): Path to the dataset configuration YAML
        results_dir (Path): Directory to save evaluation results
    """
    
    def __init__(
        self,
        model_path: str = "models/yolo_best.pt",
        data_config: str = "data/yolo_dataset/data.yaml",
        results_dir: str = "results"
    ):
        """
        Initialize the YOLO evaluator.
        
        Args:
            model_path: Path to the trained YOLO model weights
            data_config: Path to the dataset configuration YAML file
            results_dir: Directory to save evaluation results
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.model_path = self.project_root / model_path
        self.data_config = self.project_root / data_config
        self.results_dir = self.project_root / results_dir
        
        # Validate paths
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Validate model file is not empty
        if self.model_path.stat().st_size == 0:
            raise ValueError(
                f"Model file is empty (0 bytes): {self.model_path}\n"
                f"Please train the model first using train_yolo.py or provide a valid trained model."
            )
        
        if not self.data_config.exists():
            raise FileNotFoundError(f"Data config not found at {self.data_config}")
        
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized YOLO evaluator with model: {self.model_path}")
    
    def load_model(self) -> YOLO:
        """
        Load the YOLO model from the specified path.
        
        Returns:
            Loaded YOLO model instance
        
        Raises:
            Exception: If model loading fails due to corruption or invalid format
        """
        try:
            model = YOLO(str(self.model_path))
            logger.info("YOLO model loaded successfully")
            return model
        except EOFError as e:
            logger.error(f"Model file is corrupted or incomplete: {self.model_path}")
            raise ValueError(
                f"Failed to load model - file appears to be corrupted.\n"
                f"Please retrain the model or restore from backup."
            ) from e
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate(
        self,
        save_json: bool = True,
        save_plots: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the YOLO model on the validation dataset.
        
        Args:
            save_json: Whether to save metrics as JSON
            save_plots: Whether to save evaluation plots
            verbose: Whether to print detailed metrics
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting YOLO model evaluation...")
        
        # Load model
        model = self.load_model()
        
        # Run validation
        try:
            metrics = model.val(
                data=str(self.data_config),
                save_json=save_json,
                plots=save_plots,
                verbose=verbose
            )
            
            # Extract key metrics
            metrics_dict = self._extract_metrics(metrics)
            
            # Save metrics to file
            if save_json:
                self._save_metrics(metrics_dict)
            
            # Log results
            self._log_metrics(metrics_dict)
            
            logger.info("Evaluation completed successfully")
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _extract_metrics(self, metrics) -> Dict[str, Any]:
        """
        Extract relevant metrics from YOLO validation results.
        
        Args:
            metrics: YOLO validation metrics object
        
        Returns:
            Dictionary of formatted metrics
        """
        metrics_dict = {
            "box": {
                "precision": float(metrics.box.p) if hasattr(metrics.box, 'p') else None,
                "recall": float(metrics.box.r) if hasattr(metrics.box, 'r') else None,
                "map50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else None,
                "map50-95": float(metrics.box.map) if hasattr(metrics.box, 'map') else None,
            },
            "speed": {
                "preprocess": float(metrics.speed.get('preprocess', 0)),
                "inference": float(metrics.speed.get('inference', 0)),
                "postprocess": float(metrics.speed.get('postprocess', 0)),
            } if hasattr(metrics, 'speed') else {},
            "model_path": str(self.model_path),
            "data_config": str(self.data_config)
        }
        return metrics_dict
    
    def _save_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics_dict: Dictionary containing evaluation metrics
        """
        output_file = self.results_dir / "yolo_metrics.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            logger.info(f"Metrics saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _log_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """
        Log evaluation metrics to console.
        
        Args:
            metrics_dict: Dictionary containing evaluation metrics
        """
        logger.info("=" * 50)
        logger.info("YOLO Evaluation Metrics")
        logger.info("=" * 50)
        
        if "box" in metrics_dict:
            box_metrics = metrics_dict["box"]
            logger.info(f"Precision:    {box_metrics.get('precision', 'N/A'):.4f}" if box_metrics.get('precision') else "Precision:    N/A")
            logger.info(f"Recall:       {box_metrics.get('recall', 'N/A'):.4f}" if box_metrics.get('recall') else "Recall:       N/A")
            logger.info(f"mAP@50:       {box_metrics.get('map50', 'N/A'):.4f}" if box_metrics.get('map50') else "mAP@50:       N/A")
            logger.info(f"mAP@50-95:    {box_metrics.get('map50-95', 'N/A'):.4f}" if box_metrics.get('map50-95') else "mAP@50-95:    N/A")
        
        if "speed" in metrics_dict and metrics_dict["speed"]:
            speed = metrics_dict["speed"]
            logger.info(f"\nInference Speed: {speed.get('inference', 'N/A'):.2f} ms")
        
        logger.info("=" * 50)


def evaluate(
    model_path: Optional[str] = None,
    data_config: Optional[str] = None,
    save_json: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to evaluate YOLO model.
    
    Args:
        model_path: Path to model weights (default: models/yolo_best.pt)
        data_config: Path to data config (default: data/yolo_dataset/data.yaml)
        save_json: Whether to save metrics as JSON
    
    Returns:
        Dictionary containing evaluation metrics
    """
    kwargs = {}
    if model_path:
        kwargs['model_path'] = model_path
    if data_config:
        kwargs['data_config'] = data_config
    
    evaluator = YOLOEvaluator(**kwargs)
    return evaluator.evaluate(save_json=save_json)


if __name__ == "__main__":
    # Run evaluation
    metrics = evaluate()
    print("\nEvaluation complete. Results saved to results/yolo_metrics.json")
