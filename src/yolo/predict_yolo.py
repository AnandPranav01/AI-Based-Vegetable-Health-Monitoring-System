"""
YOLO Detection Module for Food Item Detection

This module provides functions and classes for performing object detection using
trained YOLO models to detect food items in images.
"""

import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
import cv2
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.utils import get_device, load_config, resolve_path, ensure_dir


class YOLOPredictor:
    """
    YOLO predictor class for efficient object detection.
    Loads model once and reuses for multiple predictions.
    
    Args:
        model_path (str or Path): Path to trained YOLO model (.pt file)
        config_path (str or Path, optional): Path to config file
        confidence_threshold (float): Minimum confidence for detections (default: 0.25)
        iou_threshold (float): IoU threshold for NMS (default: 0.45)
        device (str, optional): Device to run inference on ('cuda' or 'cpu')
        image_size (int): Input image size (default: 640)
    
    Example:
        >>> predictor = YOLOPredictor('models/yolo_best.pt')
        >>> detections = predictor.predict('test_image.jpg')
        >>> print(f"Found {len(detections)} objects")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        image_size: int = 640
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        
        # Validate model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load config if provided
        self.config = None
        if config_path:
            self.config = load_config(config_path)
            if 'yolo_detection' in self.config:
                yolo_cfg = self.config['yolo_detection']
                self.confidence_threshold = yolo_cfg.get('confidence_threshold', confidence_threshold)
                self.iou_threshold = yolo_cfg.get('iou_threshold', iou_threshold)
                self.image_size = yolo_cfg.get('image_size', image_size)
                device = device or yolo_cfg.get('device', None)
        
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load YOLO model
        self.model = self._load_model()
        
        # Get class names
        self.class_names = self.model.names
    
    def _load_model(self) -> YOLO:
        """Load the trained YOLO model."""
        try:
            model = YOLO(str(self.model_path))
            
            # Move to specified device
            if hasattr(model, 'to'):
                model.to(self.device)
            
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {e}")
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        save_results: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        return_format: str = 'dict'
    ) -> Union[List[Dict[str, Any]], Any]:
        """
        Detect objects in a single image.
        
        Args:
            image (str, Path, np.ndarray, or PIL.Image): Input image
            save_results (bool): Whether to save visualization
            save_dir (str or Path, optional): Directory to save results
            return_format (str): Format of return value ('dict', 'raw', 'boxes')
        
        Returns:
            list or Results: Detection results
                If return_format='dict', returns list of dicts with keys:
                    - 'bbox': [x1, y1, x2, y2]
                    - 'confidence': float
                    - 'class_id': int
                    - 'class_name': str
                If return_format='raw', returns ultralytics Results object
                If return_format='boxes', returns only bbox coordinates
        
        Example:
            >>> detections = predictor.predict('image.jpg')
            >>> for det in detections:
            ...     print(f"{det['class_name']}: {det['confidence']:.2f}")
        """
        # Handle save directory
        if save_results and save_dir:
            save_dir = ensure_dir(save_dir)
        
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            save=save_results,
            project=str(save_dir) if save_dir else None
        )
        
        # Return based on format
        if return_format == 'raw':
            return results
        elif return_format == 'dict':
            return self._parse_results(results[0])
        elif return_format == 'boxes':
            return self._extract_boxes(results[0])
        else:
            raise ValueError(f"Invalid return_format: {return_format}")
    
    def _parse_results(self, result) -> List[Dict[str, Any]]:
        """
        Parse YOLO results into structured dictionary format.
        
        Args:
            result: Single YOLO result object
        
        Returns:
            list: List of detection dictionaries
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for i in range(len(boxes)):
            detection = {
                'bbox': boxes[i].tolist(),
                'confidence': float(confidences[i]),
                'class_id': int(class_ids[i]),
                'class_name': self.class_names[class_ids[i]],
                'bbox_xyxy': boxes[i].tolist(),  # [x1, y1, x2, y2]
                'bbox_xywh': self._xyxy_to_xywh(boxes[i]).tolist()  # [x_center, y_center, w, h]
            }
            detections.append(detection)
        
        return detections
    
    def _extract_boxes(self, result) -> np.ndarray:
        """Extract only bounding box coordinates."""
        if result.boxes is None or len(result.boxes) == 0:
            return np.array([])
        return result.boxes.xyxy.cpu().numpy()
    
    @staticmethod
    def _xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
        """Convert bbox from [x1, y1, x2, y2] to [x_center, y_center, w, h]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x_center = x1 + w / 2
        y_center = y1 + h / 2
        return np.array([x_center, y_center, w, h])
    
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        save_results: bool = False,
        save_dir: Optional[Union[str, Path]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect objects in multiple images.
        
        Args:
            images (list): List of image paths or arrays
            save_results (bool): Whether to save visualizations
            save_dir (str or Path, optional): Directory to save results
        
        Returns:
            list: List of detection lists (one per image)
        
        Example:
            >>> images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
            >>> batch_results = predictor.predict_batch(images)
        """
        if save_results and save_dir:
            save_dir = ensure_dir(save_dir)
        
        # Run batch inference
        results = self.model(
            images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            save=save_results,
            project=str(save_dir) if save_dir else None
        )
        
        # Parse all results
        return [self._parse_results(result) for result in results]
    
    def predict_and_crop(
        self,
        image: Union[str, Path, np.ndarray],
        padding: int = 0,
        min_size: int = 50
    ) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        """
        Detect objects and return cropped regions (useful for pipeline).
        
        Args:
            image (str, Path, or np.ndarray): Input image
            padding (int): Padding pixels around crops
            min_size (int): Minimum crop size in pixels
        
        Returns:
            tuple: (detections, crops)
                - detections: List of detection dictionaries
                - crops: List of cropped image arrays
        
        Example:
            >>> detections, crops = predictor.predict_and_crop('image.jpg', padding=10)
            >>> # Now pass crops to ResNet classifier
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            img_array = cv2.imread(str(image_path))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            img_array = image
        
        # Get detections
        detections = self.predict(img_array, return_format='dict')
        
        # Crop detected regions
        crops = []
        h, w = img_array.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Check minimum size
            if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
                crop = img_array[y1:y2, x1:x2]
                crops.append(crop)
            else:
                crops.append(None)  # Placeholder for too-small detections
        
        return detections, crops
    
    def visualize_results(
        self,
        image: Union[str, Path, np.ndarray],
        detections: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image (str, Path, or np.ndarray): Input image
            detections (list): List of detection dictionaries
            output_path (str or Path, optional): Path to save visualization
            thickness (int): Box line thickness
            font_scale (float): Font scale for labels
        
        Returns:
            np.ndarray: Annotated image
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), thickness)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), img)
        
        return img
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'image_size': self.image_size,
            'num_classes': len(self.class_names),
            'class_names': self.class_names
        }


# ============================================================================
# Standalone Functions (for simple use cases)
# ============================================================================

def predict(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = "models/yolo_best.pt",
    save_results: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    confidence: float = 0.25,
    device: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Simple detection function for a single image.
    
    Args:
        image_path (str or Path): Path to image file
        model_path (str or Path): Path to trained YOLO model
        save_results (bool): Whether to save visualization
        save_dir (str or Path, optional): Directory to save results
        confidence (float): Confidence threshold
        device (str, optional): Device to use ('cuda' or 'cpu')
    
    Returns:
        list: List of detection dictionaries
    
    Example:
        >>> detections = predict('test.jpg')
        >>> print(f"Found {len(detections)} objects")
        >>> for det in detections:
        ...     print(f"{det['class_name']}: {det['confidence']:.2f}")
    """
    # Resolve paths
    model_path = resolve_path(model_path)
    image_path = resolve_path(image_path)
    
    # Create predictor and run detection
    predictor = YOLOPredictor(
        model_path,
        confidence_threshold=confidence,
        device=device
    )
    
    return predictor.predict(
        image_path,
        save_results=save_results,
        save_dir=save_dir,
        return_format='dict'
    )


def predict_with_config(
    image_path: Union[str, Path],
    config_path: Union[str, Path] = "configs/yolo_config.yaml",
    model_path: Optional[Union[str, Path]] = None,
    save_results: bool = False
) -> List[Dict[str, Any]]:
    """
    Detect objects using configuration file.
    
    Args:
        image_path (str or Path): Path to image file
        config_path (str or Path): Path to config file
        model_path (str or Path, optional): Override model path from config
        save_results (bool): Whether to save visualization
    
    Returns:
        list: List of detection dictionaries
    
    Example:
        >>> detections = predict_with_config('test.jpg', 'configs/yolo_config.yaml')
    """
    config = load_config(config_path)
    
    if model_path is None:
        # Try to get from config
        if 'model' in config:
            model_path = config['model']
        else:
            model_path = 'models/yolo_best.pt'
    
    predictor = YOLOPredictor(model_path, config_path=config_path)
    
    save_dir = None
    if save_results:
        save_dir = config.get('project', 'runs/detect')
    
    return predictor.predict(
        image_path,
        save_results=save_results,
        save_dir=save_dir,
        return_format='dict'
    )


def detect_and_crop(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = "models/yolo_best.pt",
    padding: int = 10,
    confidence: float = 0.25
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Detect objects and return crops (convenient for pipeline integration).
    
    Args:
        image_path (str or Path): Path to image file
        model_path (str or Path): Path to YOLO model
        padding (int): Padding around crops
        confidence (float): Confidence threshold
    
    Returns:
        tuple: (detections, crops)
    
    Example:
        >>> detections, crops = detect_and_crop('image.jpg', padding=10)
        >>> # Pass crops to ResNet classifier
    """
    predictor = YOLOPredictor(model_path, confidence_threshold=confidence)
    return predictor.predict_and_crop(image_path, padding=padding)


# ============================================================================
# Main / Testing
# ============================================================================

def main():
    """Main function for testing YOLO detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Food Item Detection')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/yolo_best.pt',
                       help='Path to YOLO model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--save', action='store_true',
                       help='Save visualization')
    parser.add_argument('--save-dir', type=str, default='runs/detect',
                       help='Directory to save results')
    parser.add_argument('--batch', nargs='+',
                       help='Multiple images for batch detection')
    parser.add_argument('--crop', action='store_true',
                       help='Extract and save crops')
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # Batch detection
            print(f"Processing {len(args.batch)} images...")
            predictor = YOLOPredictor(
                args.model,
                config_path=args.config,
                confidence_threshold=args.conf,
                device=args.device
            )
            batch_results = predictor.predict_batch(
                args.batch,
                save_results=args.save,
                save_dir=args.save_dir if args.save else None
            )
            
            for img_path, detections in zip(args.batch, batch_results):
                print(f"\n{img_path}: {len(detections)} detections")
                for det in detections:
                    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
        
        elif args.crop:
            # Detection with crops
            detections, crops = detect_and_crop(
                args.image,
                args.model,
                confidence=args.conf
            )
            
            print(f"Image: {args.image}")
            print(f"Detections: {len(detections)}")
            
            # Save crops
            crop_dir = Path(args.save_dir) / 'crops'
            crop_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (det, crop) in enumerate(zip(detections, crops)):
                if crop is not None:
                    crop_path = crop_dir / f"crop_{i}_{det['class_name']}.jpg"
                    # Convert RGB to BGR for OpenCV
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(crop_path), crop_bgr)
                    print(f"  - {det['class_name']}: {det['confidence']:.2f} -> {crop_path}")
        
        else:
            # Single image detection
            if args.config:
                detections = predict_with_config(
                    args.image,
                    args.config,
                    args.model,
                    save_results=args.save
                )
            else:
                detections = predict(
                    args.image,
                    args.model,
                    save_results=args.save,
                    save_dir=args.save_dir if args.save else None,
                    confidence=args.conf,
                    device=args.device
                )
            
            print(f"Image: {args.image}")
            print(f"Detections: {len(detections)}")
            
            for det in detections:
                bbox = det['bbox']
                print(f"  - {det['class_name']}: {det['confidence']:.2%} "
                      f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
            
            if args.save:
                print(f"\nResults saved to: {args.save_dir}")
    
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
            detections = predict("sample.jpg", save_results=True)
            print(f"Found {len(detections)} objects:")
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.2f}")
        except FileNotFoundError:
            print("Test image 'sample.jpg' not found. Usage:")
            print("  python predict_yolo.py <image_path>")
            print("  python predict_yolo.py image.jpg --model models/yolo_best.pt")
            print("  python predict_yolo.py image.jpg --save --conf 0.5")
            print("  python predict_yolo.py --batch img1.jpg img2.jpg img3.jpg")
            print("  python predict_yolo.py image.jpg --crop  # Extract detected regions")
