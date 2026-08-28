"""
ResNet Prediction Module for Food Spoilage Classification

This module provides functions and classes for performing inference using
trained ResNet models to classify food items as fresh or spoiled.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Union, List, Tuple, Dict, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from resnet.model import build_resnet, load_model
from pipeline.utils import get_device, load_config, resolve_path


# Default ImageNet normalization for pretrained ResNet models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class labels
CLASS_LABELS = {
    0: 'fresh',
    1: 'spoiled'
}


class ResNetPredictor:
    """
    ResNet predictor class for efficient inference.
    Loads model once and reuses for multiple predictions.
    
    Args:
        model_path (str or Path): Path to trained model checkpoint
        config_path (str or Path, optional): Path to config file
        device (str, optional): Device to run inference on ('cuda' or 'cpu')
        image_size (int): Input image size (default: 224)
    
    Example:
        >>> predictor = ResNetPredictor('models/resnet_spoilage.pt')
        >>> result = predictor.predict('test_image.jpg')
        >>> print(f"Class: {result['class']}, Confidence: {result['confidence']:.2f}")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        image_size: int = 224
    ):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.device = get_device(device, verbose=False)
        
        # Load config if provided
        self.config = None
        if config_path:
            self.config = load_config(config_path)
            if 'resnet_classification' in self.config:
                self.image_size = self.config['resnet_classification'].get('image_size', 224)
        
        # Validate model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup preprocessing transform
        self.transform = self._create_transform()
    
    def _load_model(self) -> torch.nn.Module:
        """Load the trained ResNet model."""
        try:
            # Try to load using the load_model function
            model = load_model(
                str(self.model_path),
                architecture='resnet50',
                num_classes=2,
                dropout=0.5,
                device=str(self.device)
            )
        except Exception as e:
            # Fallback: build model and load state dict
            model = build_resnet(num_classes=2, pretrained=False)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
        
        return model
    
    def _create_transform(self) -> transforms.Compose:
        """Create image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and validate image file.
        
        Args:
            image_path (str or Path): Path to image file
        
        Returns:
            PIL.Image: Loaded image in RGB format
        
        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If file is not a valid image
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    def _preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image (str, Path, or PIL.Image): Input image
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        return self.transform(image).unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        return_confidence: bool = True
    ) -> Dict[str, Union[str, float, int]]:
        """
        Predict food spoilage for a single image.
        
        Args:
            image (str, Path, or PIL.Image): Input image
            return_confidence (bool): Whether to return confidence scores
        
        Returns:
            dict: Prediction results containing:
                - 'class': Predicted class name ('fresh' or 'spoiled')
                - 'class_id': Class ID (0 or 1)
                - 'confidence': Confidence score (if return_confidence=True)
                - 'probabilities': Class probabilities (if return_confidence=True)
        
        Example:
            >>> result = predictor.predict('apple.jpg')
            >>> print(f"{result['class']}: {result['confidence']:.1%}")
            fresh: 95.3%
        """
        # Preprocess image
        img_tensor = self._preprocess_image(image)
        
        # Get model output
        output = self.model(img_tensor)
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=1)
        
        # Get prediction
        class_id = output.argmax(1).item()
        class_name = CLASS_LABELS[class_id]
        confidence = probabilities[0, class_id].item()
        
        # Build result dictionary
        result = {
            'class': class_name,
            'class_id': class_id
        }
        
        if return_confidence:
            result['confidence'] = confidence
            result['probabilities'] = {
                'fresh': probabilities[0, 0].item(),
                'spoiled': probabilities[0, 1].item()
            }
        
        # Freshness percentage: 0.0 (fully spoiled) to 100.0 (fully fresh)
        result['freshness_percentage'] = round(probabilities[0, 0].item() * 100, 2)
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 8
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        Predict food spoilage for multiple images.
        
        Args:
            images (list): List of image paths or PIL images
            batch_size (int): Batch size for processing
        
        Returns:
            list: List of prediction dictionaries
        
        Example:
            >>> images = ['apple1.jpg', 'apple2.jpg', 'banana.jpg']
            >>> results = predictor.predict_batch(images)
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = [self._preprocess_image(img) for img in batch]
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Get predictions
            outputs = self.model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Extract results
            for j in range(len(batch)):
                class_id = outputs[j].argmax().item()
                confidence = probabilities[j, class_id].item()
                
                results.append({
                    'class': CLASS_LABELS[class_id],
                    'class_id': class_id,
                    'confidence': confidence,
                    'probabilities': {
                        'fresh': probabilities[j, 0].item(),
                        'spoiled': probabilities[j, 1].item()
                    }
                })
        
        return results
    
    def predict_from_crop(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        padding: int = 0
    ) -> Dict[str, Union[str, float, int]]:
        """
        Predict from image array or cropped region (useful for pipeline integration).
        
        Args:
            image (np.ndarray): Image array (H, W, C) in RGB or BGR format
            bbox (tuple, optional): Bounding box (x1, y1, x2, y2) for cropping
            padding (int): Padding to add around bbox
        
        Returns:
            dict: Prediction results
        
        Example:
            >>> # From YOLO detection
            >>> crop = image[y1:y2, x1:x2]
            >>> result = predictor.predict_from_crop(crop)
        """
        # Crop if bbox provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            image = image[y1:y2, x1:x2]
        
        # Convert numpy array to PIL Image.
        # Input MUST be RGB — the pipeline converts BGR→RGB before cropping,
        # matching the RGB colour space used by PIL/ImageFolder during training.
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        else:
            raise ValueError("Image must be uint8 numpy array")
        
        return self.predict(pil_image)


# ============================================================================
# Standalone Functions (for simple use cases)
# ============================================================================

def predict(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = "models/resnet_spoilage.pt",
    device: Optional[str] = None,
    return_confidence: bool = True
) -> Union[str, Dict[str, Union[str, float]]]:
    """
    Simple prediction function for a single image.
    
    Args:
        image_path (str or Path): Path to image file
        model_path (str or Path): Path to trained model
        device (str, optional): Device to use ('cuda' or 'cpu')
        return_confidence (bool): Return full result dict or just class name
    
    Returns:
        str or dict: Predicted class ('fresh' or 'spoiled') or full result dict
    
    Example:
        >>> result = predict('test_apple.jpg')
        >>> print(result)
        'fresh'
        
        >>> result = predict('test_apple.jpg', return_confidence=True)
        >>> print(f"{result['class']}: {result['confidence']:.2%}")
        fresh: 92.5%
    """
    # Resolve paths
    model_path = resolve_path(model_path)
    image_path = resolve_path(image_path)
    
    # Create predictor and run prediction
    predictor = ResNetPredictor(model_path, device=device)
    result = predictor.predict(image_path, return_confidence=return_confidence)
    
    if return_confidence:
        return result
    else:
        return result['class']


def predict_with_config(
    image_path: Union[str, Path],
    config_path: Union[str, Path] = "configs/resnet_config.yaml",
    model_path: Optional[Union[str, Path]] = None
) -> Dict[str, Union[str, float]]:
    """
    Predict using configuration file.
    
    Args:
        image_path (str or Path): Path to image file
        config_path (str or Path): Path to config file
        model_path (str or Path, optional): Override model path from config
    
    Returns:
        dict: Prediction results
    
    Example:
        >>> result = predict_with_config('test.jpg', 'configs/resnet_config.yaml')
    """
    config = load_config(config_path)
    
    if model_path is None:
        model_path = config.get('checkpoint', {}).get('save_dir', 'models')
        model_path = Path(model_path) / 'resnet_spoilage.pt'
    
    device = config.get('device', 'cpu')
    
    predictor = ResNetPredictor(model_path, config_path=config_path, device=device)
    return predictor.predict(image_path)


# ============================================================================
# Main / Testing
# ============================================================================

def main():
    """Main function for testing predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ResNet Food Spoilage Prediction')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/resnet_spoilage.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--batch', nargs='+', help='Multiple images for batch prediction')
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # Batch prediction
            print(f"Processing {len(args.batch)} images...")
            predictor = ResNetPredictor(args.model, config_path=args.config, device=args.device)
            results = predictor.predict_batch(args.batch)
            
            for img_path, result in zip(args.batch, results):
                print(f"{img_path}: {result['class']} ({result['confidence']:.1%})")
        else:
            # Single prediction
            if args.config:
                result = predict_with_config(args.image, args.config, args.model)
            else:
                result = predict(args.image, args.model, args.device, return_confidence=True)
            
            print(f"Image: {args.image}")
            print(f"Prediction: {result['class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Probabilities: Fresh={result['probabilities']['fresh']:.2%}, "
                  f"Spoiled={result['probabilities']['spoiled']:.2%}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Simple test mode
    import sys
    
    if len(sys.argv) > 1:
        exit(main())
    else:
        # Default test
        try:
            result = predict("crop.jpg")
            print(f"Result: {result}")
        except FileNotFoundError:
            print("Test image 'crop.jpg' not found. Usage:")
            print("  python predict_resnet.py <image_path>")
            print("  python predict_resnet.py image.jpg --model models/resnet_spoilage.pt")
            print("  python predict_resnet.py --batch img1.jpg img2.jpg img3.jpg")
