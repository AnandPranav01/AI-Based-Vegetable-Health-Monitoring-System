"""
ResNet Evaluation Module for Food Spoilage Classification

This module provides comprehensive evaluation functionality including
accuracy, precision, recall, F1-score, confusion matrix, and per-class metrics.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import sys
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from resnet.model import build_resnet, load_model, build_resnet_from_config
from pipeline.utils import (
    load_config, save_json, setup_logger, get_device, 
    ensure_dir, resolve_path
)


class ResNetEvaluator:
    """
    Evaluator class for ResNet models with comprehensive metrics.
    
    Args:
        model_path (str or Path): Path to trained model checkpoint
        config_path (str or Path, optional): Path to configuration file
        device (str, optional): Device to run evaluation on
        dataset_split (str): Dataset split to evaluate ('val' or 'test')
    
    Example:
        >>> evaluator = ResNetEvaluator('models/resnet_spoilage.pt')
        >>> results = evaluator.evaluate()
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        dataset_split: str = 'val'
    ):
        self.model_path = Path(model_path)
        self.dataset_split = dataset_split
        
        # Validate model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load configuration
        self.config = None
        if config_path:
            self.config = load_config(config_path)
        
        # Setup device
        if device:
            self.device = get_device(device, verbose=False)
        elif self.config:
            self.device = get_device(self.config.get('device', 'cuda'), verbose=False)
        else:
            self.device = get_device(verbose=False)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup data loader
        self.data_loader, self.class_names = self._setup_data_loader()
        
        # Class labels
        self.class_labels = {i: name for i, name in enumerate(self.class_names)}
    
    def _load_model(self) -> nn.Module:
        """Load the trained model."""
        try:
            if self.config:
                # Load with config
                model = build_resnet_from_config(self.config)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Load without config
                model = load_model(
                    str(self.model_path),
                    architecture='resnet50',
                    num_classes=2,
                    dropout=0.5,
                    device=str(self.device)
                )
            
            model = model.to(self.device)
            print(f"Model loaded from: {self.model_path}")
            return model
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _setup_data_loader(self) -> Tuple[DataLoader, List[str]]:
        """Setup data loader for evaluation."""
        # Get data directory
        if self.config:
            data_config = self.config.get('data', {})
            if self.dataset_split == 'test':
                data_dir = data_config.get('test_dir', 'data/resnet_dataset/test')
            else:
                data_dir = data_config.get('val_dir', 'data/resnet_dataset/val')
            
            image_size = data_config.get('image_size', 224)
            batch_size = self.config.get('training', {}).get('batch_size', 32)
            num_workers = data_config.get('num_workers', 4)
            
            # Get normalization settings
            aug_config = self.config.get('augmentation', {})
            normalize_config = aug_config.get('normalize', {})
            mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
            std = normalize_config.get('std', [0.229, 0.224, 0.225])
        else:
            data_dir = f'data/resnet_dataset/{self.dataset_split}'
            image_size = 224
            batch_size = 32
            num_workers = 4
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        
        data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        # Create transform
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Load dataset
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        
        print(f"Evaluating on {self.dataset_split} set")
        print(f"Dataset: {data_dir}")
        print(f"Samples: {len(dataset)}")
        print(f"Classes: {dataset.classes}")
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return data_loader, dataset.classes
    
    @torch.no_grad()
    def evaluate(self, save_results: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate model and return comprehensive metrics.
        
        Args:
            save_results (bool): Whether to save results to file
            output_dir (str, optional): Directory to save results
        
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall, F1, etc.
        """
        print("\n" + "="*60)
        print("Starting Evaluation")
        print("="*60 + "\n")
        
        # Collect predictions and labels
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        self.model.eval()
        
        for images, labels in tqdm(self.data_loader, desc="Evaluating"):
            images = images.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels, all_probabilities)
        
        # Print results
        self._print_results(metrics)
        
        # Save results if requested
        if save_results:
            if output_dir is None:
                output_dir = 'results'
            self._save_results(metrics, output_dir)
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Overall metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Per-class metrics (use 'macro' for balanced classes, 'weighted' for imbalanced)
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        
        precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(labels, predictions)
        
        # Per-class accuracy
        per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        # Class distribution (convert numpy int64 keys/values to plain Python int)
        unique, counts = np.unique(labels, return_counts=True)
        class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
        
        # Organize metrics
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
            },
            'per_class': {},
            'confusion_matrix': conf_matrix.tolist(),
            'class_distribution': class_distribution,
            'total_samples': len(labels),
            'model_path': str(self.model_path),
            'dataset_split': self.dataset_split
        }
        
        # Per-class detailed metrics
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'accuracy': float(per_class_accuracy[i]),
                'support': int(class_distribution.get(i, 0))
            }
        
        return metrics
    
    def _print_results(self, metrics: Dict[str, Any]):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Overall metrics
        print("\n📊 Overall Metrics:")
        print(f"  Accuracy:          {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)")
        print(f"  Precision (macro): {metrics['overall']['precision_macro']:.4f}")
        print(f"  Recall (macro):    {metrics['overall']['recall_macro']:.4f}")
        print(f"  F1-Score (macro):  {metrics['overall']['f1_macro']:.4f}")
        
        # Per-class metrics
        print("\n📈 Per-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12} {'Support':<10}")
        print("-" * 75)
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<15} "
                  f"{class_metrics['precision']:.4f}       "
                  f"{class_metrics['recall']:.4f}       "
                  f"{class_metrics['f1_score']:.4f}       "
                  f"{class_metrics['accuracy']:.4f}       "
                  f"{class_metrics['support']:<10}")
        
        # Confusion matrix
        print("\n🔍 Confusion Matrix:")
        conf_matrix = np.array(metrics['confusion_matrix'])
        
        # Header
        print(f"{'':>15} ", end="")
        for class_name in self.class_names:
            print(f"{class_name:>10}", end=" ")
        print()
        
        # Rows
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:>15} ", end="")
            for j in range(len(self.class_names)):
                print(f"{conf_matrix[i][j]:>10}", end=" ")
            print()
        
        print("\n" + "="*60 + "\n")
    
    def _save_results(self, metrics: Dict[str, Any], output_dir: str):
        """Save evaluation results to JSON file."""
        output_dir = ensure_dir(output_dir)
        
        # Save metrics
        metrics_file = output_dir / f'resnet_metrics_{self.dataset_split}.json'
        save_json(metrics, metrics_file)
        print(f"✅ Metrics saved to: {metrics_file}")
        
        # Save confusion matrix as separate file for easy plotting
        conf_matrix_data = {
            'confusion_matrix': metrics['confusion_matrix'],
            'class_names': self.class_names
        }
        conf_matrix_file = output_dir / f'confusion_matrix_{self.dataset_split}.json'
        save_json(conf_matrix_data, conf_matrix_file)
        print(f"✅ Confusion matrix saved to: {conf_matrix_file}")
    
    def evaluate_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Evaluate a single image (useful for testing).
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            dict: Prediction results
        """
        from PIL import Image
        
        # Load and preprocess image
        if self.config:
            aug_config = self.config.get('augmentation', {})
            normalize_config = aug_config.get('normalize', {})
            mean = normalize_config.get('mean', [0.485, 0.456, 0.406])
            std = normalize_config.get('std', [0.229, 0.224, 0.225])
            image_size = self.config.get('data', {}).get('image_size', 224)
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image_size = 224
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(1).item()
        
        return {
            'image_path': str(image_path),
            'predicted_class': self.class_names[prediction],
            'predicted_class_id': prediction,
            'confidence': probabilities[0, prediction].item(),
            'probabilities': {
                self.class_names[i]: probabilities[0, i].item()
                for i in range(len(self.class_names))
            }
        }


# ============================================================================
# Standalone Evaluation Functions
# ============================================================================

def evaluate_resnet(
    model_path: str = 'models/resnet_spoilage.pt',
    config_path: Optional[str] = None,
    dataset_split: str = 'val',
    save_results: bool = True,
    output_dir: str = 'results',
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate ResNet model and return metrics.
    
    Args:
        model_path (str): Path to trained model
        config_path (str, optional): Path to config file
        dataset_split (str): Dataset split ('val' or 'test')
        save_results (bool): Whether to save results
        output_dir (str): Directory to save results
        device (str, optional): Device to use
    
    Returns:
        dict: Evaluation metrics
    
    Example:
        >>> metrics = evaluate_resnet('models/resnet_spoilage.pt', 
        ...                          config_path='configs/resnet_config.yaml')
        >>> print(f"Accuracy: {metrics['overall']['accuracy']:.2%}")
    """
    evaluator = ResNetEvaluator(
        model_path=model_path,
        config_path=config_path,
        device=device,
        dataset_split=dataset_split
    )
    
    return evaluator.evaluate(save_results=save_results, output_dir=output_dir)


def compare_models(
    model_paths: List[str],
    config_path: Optional[str] = None,
    dataset_split: str = 'val',
    output_dir: str = 'results'
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        model_paths (list): List of model checkpoint paths
        config_path (str, optional): Path to config file
        dataset_split (str): Dataset split to evaluate on
        output_dir (str): Directory to save comparison results
    
    Returns:
        dict: Comparison results for all models
    
    Example:
        >>> models = ['models/resnet50.pt', 'models/resnet101.pt']
        >>> comparison = compare_models(models)
    """
    results = {}
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60 + "\n")
    
    for model_path in model_paths:
        print(f"\nEvaluating: {model_path}")
        
        evaluator = ResNetEvaluator(
            model_path=model_path,
            config_path=config_path,
            dataset_split=dataset_split
        )
        
        metrics = evaluator.evaluate(save_results=False)
        results[str(model_path)] = metrics
    
    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60 + "\n")
    print(f"{'Model':<40} {'Accuracy':<12} {'F1 (macro)':<12}")
    print("-" * 65)
    
    for model_path, metrics in results.items():
        print(f"{Path(model_path).name:<40} "
              f"{metrics['overall']['accuracy']:.4f}      "
              f"{metrics['overall']['f1_macro']:.4f}")
    
    # Save comparison
    output_dir = ensure_dir(output_dir)
    comparison_file = output_dir / 'model_comparison.json'
    save_json(results, comparison_file)
    print(f"\n✅ Comparison saved to: {comparison_file}\n")
    
    return results


# ============================================================================
# Main / CLI
# ============================================================================

def main():
    """Main function for CLI evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ResNet for Food Spoilage Detection')
    parser.add_argument('--model', type=str, default='models/resnet_spoilage.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                       help='Dataset split to evaluate (val or test)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple models (provide multiple paths)')
    parser.add_argument('--image', type=str, default=None,
                       help='Evaluate single image (test mode)')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Compare multiple models
            compare_models(
                model_paths=args.compare,
                config_path=args.config,
                dataset_split=args.split,
                output_dir=args.output
            )
        
        elif args.image:
            # Single image evaluation
            evaluator = ResNetEvaluator(
                model_path=args.model,
                config_path=args.config,
                device=args.device
            )
            result = evaluator.evaluate_single_image(args.image)
            
            print(f"\nImage: {result['image_path']}")
            print(f"Prediction: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nProbabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.2%}")
        
        else:
            # Standard evaluation
            metrics = evaluate_resnet(
                model_path=args.model,
                config_path=args.config,
                dataset_split=args.split,
                save_results=not args.no_save,
                output_dir=args.output,
                device=args.device
            )
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
