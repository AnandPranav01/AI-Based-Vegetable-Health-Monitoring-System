"""
Two-Stage Pipeline Inference for Food Spoilage Detection

This module implements the complete detection and classification pipeline:
Stage 1: YOLO detects food items in images
Stage 2: ResNet classifies each detected item as fresh or spoiled

Features:
- Config-driven pipeline
- Batch processing
- Visualization with bounding boxes
- Comprehensive output with confidence scores
- Memory-efficient (no temp files)
- Error handling and logging
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import sys
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from yolo.predict_yolo import YOLOPredictor
from resnet.predict_resnet import ResNetPredictor
from pipeline.utils import (
    load_config, save_json, setup_logger, ensure_dir,
    get_device, resolve_path, Timer, get_timestamp
)


class FoodSpoilagePipeline:
    """
    Complete two-stage pipeline for food spoilage detection.
    
    Stage 1: YOLO detects food items in image
    Stage 2: ResNet classifies each detection as fresh/spoiled
    
    Args:
        config_path (str or Path, optional): Path to pipeline config file
        yolo_model_path (str or Path, optional): Path to YOLO model
        resnet_model_path (str or Path, optional): Path to ResNet model
        device (str, optional): Device to use ('cuda' or 'cpu')
    
    Example:
        >>> pipeline = FoodSpoilagePipeline('configs/pipeline_config.yaml')
        >>> results = pipeline.process('image.jpg', save_visualization=True)
        >>> print(f"Detected {len(results['detections'])} items")
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        yolo_model_path: Optional[Union[str, Path]] = None,
        resnet_model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Get model paths
        if yolo_model_path is None:
            yolo_model_path = self.config.get('models', {}).get('yolo', {}).get('path', 'models/yolo_best.pt')
        if resnet_model_path is None:
            resnet_model_path = self.config.get('models', {}).get('resnet', {}).get('path', 'models/resnet_spoilage.pt')
        
        # Setup device
        if device is None:
            device = self.config.get('yolo_detection', {}).get('device', 'cuda')
        
        self.device = get_device(device, verbose=False)
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("Initializing Food Spoilage Detection Pipeline")
        self.logger.info(f"Device: {self.device}")
        
        # Initialize YOLO detector (Stage 1)
        self.logger.info("Loading YOLO model...")
        self.yolo_predictor = YOLOPredictor(
            model_path=yolo_model_path,
            config_path=config_path,
            device=str(self.device)
        )
        self.logger.info(f"YOLO model loaded: {yolo_model_path}")
        
        # Initialize ResNet classifier (Stage 2)
        self.logger.info("Loading ResNet model...")
        self.resnet_predictor = ResNetPredictor(
            model_path=resnet_model_path,
            config_path=config_path,
            device=str(self.device)
        )
        self.logger.info(f"ResNet model loaded: {resnet_model_path}")
        
        # Get pipeline settings
        self.pipeline_config = self.config.get('pipeline', {})
        self.output_config = self.config.get('output', {})
        self.post_processing_config = self.config.get('post_processing', {})
        
        self.logger.info("Pipeline initialization complete")
    
    def _default_config(self) -> Dict[str, Any]:
        """Create default configuration if none provided."""
        return {
            'models': {
                'yolo': {'path': 'models/yolo_best.pt'},
                'resnet': {'path': 'models/resnet_spoilage.pt'}
            },
            'yolo_detection': {
                'confidence_threshold': 0.25,
                'device': 'cuda'
            },
            'resnet_classification': {
                'confidence_threshold': 0.5
            },
            'pipeline': {
                'crop_padding': 10,
                'min_crop_size': 50
            },
            'output': {
                'save_dir': 'results/pipeline_output',
                'visualization': {
                    'bbox_thickness': 2,
                    'font_scale': 0.6,
                    'colors': {
                        'fresh': [0, 255, 0],
                        'spoiled': [0, 0, 255]
                    }
                }
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        
        if log_config.get('enabled', True):
            log_dir = Path(log_config.get('log_dir', 'logs/pipeline_logs'))
            log_file = log_dir / log_config.get('log_file', 'pipeline.log')
            
            self.logger = setup_logger(
                name='pipeline',
                log_file=log_file,
                level=log_config.get('level', 'INFO'),
                console_output=log_config.get('console_output', True)
            )
        else:
            import logging
            self.logger = logging.getLogger('pipeline')
            self.logger.setLevel(logging.INFO)
    
    def process_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        save_visualization: bool = False,
        save_crops: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        return_crops: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image (str, Path, np.ndarray, or PIL.Image): Input image
            save_visualization (bool): Save annotated image
            save_crops (bool): Save individual crops
            output_dir (str or Path, optional): Custom output directory
            return_crops (bool): Include crops in return value
        
        Returns:
            dict: Processing results with detections and classifications
                {
                    'detections': [...],  # List of detection+classification results
                    'summary': {...},     # Summary statistics
                    'metadata': {...},    # Processing metadata
                    'visualization_path': str,  # Path to saved visualization (if saved)
                    'crops': [...]        # Crop images (if return_crops=True)
                }
        
        Example:
            >>> results = pipeline.process_image('food.jpg', save_visualization=True)
            >>> for det in results['detections']:
            ...     print(f"{det['object_class']}: {det['spoilage_status']}")
        """
        start_time = time.time()
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image_name = image_path.stem
            img_array = cv2.imread(str(image_path))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            image_name = f"image_{get_timestamp()}"
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
        
        self.logger.info(f"Processing image: {image_name}")
        
        # Stage 1: YOLO Detection
        self.logger.info("Stage 1: Detecting food items with YOLO...")
        with Timer("YOLO Detection", verbose=False) as yolo_timer:
            yolo_detections = self.yolo_predictor.predict(img_array, return_format='dict')
        
        self.logger.info(f"YOLO detected {len(yolo_detections)} objects in {yolo_timer.elapsed:.2f}s")
        
        if len(yolo_detections) == 0:
            self.logger.info("No objects detected. Pipeline complete.")
            return {
                'detections': [],
                'summary': {
                    'total_detections': 0,
                    'fresh_count': 0,
                    'spoiled_count': 0
                },
                'metadata': {
                    'image_name': image_name,
                    'processing_time': time.time() - start_time,
                    'yolo_time': yolo_timer.elapsed,
                    'resnet_time': 0.0
                }
            }
        
        # Stage 2: ResNet Classification
        self.logger.info("Stage 2: Classifying detected items with ResNet...")
        
        padding = self.pipeline_config.get('crop_padding', 10)
        min_size = self.pipeline_config.get('min_crop_size', 50)
        
        detections = []
        crops_list = []
        
        with Timer("ResNet Classification", verbose=False) as resnet_timer:
            for i, det in enumerate(yolo_detections):
                # Extract bounding box
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                
                # Add padding
                h, w = img_array.shape[:2]
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                # Check minimum size
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    self.logger.warning(f"Detection {i} too small, skipping classification")
                    continue
                
                # Crop region
                crop = img_array[y1:y2, x1:x2]
                
                # Classify with ResNet
                classification = self.resnet_predictor.predict_from_crop(crop)
                
                # Combine YOLO and ResNet results
                combined_result = {
                    'detection_id': i,
                    'bbox': det['bbox'],
                    'bbox_padded': [x1, y1, x2, y2],
                    'object_class': det['class_name'],
                    'object_confidence': det['confidence'],
                    'spoilage_status': classification['class'],
                    'spoilage_confidence': classification['confidence'],
                    'spoilage_probabilities': classification['probabilities'],
                    'freshness_percentage': classification['freshness_percentage'],
                    'combined_confidence': (det['confidence'] + classification['confidence']) / 2
                }
                
                detections.append(combined_result)
                
                if return_crops or save_crops:
                    crops_list.append(crop)
        
        self.logger.info(f"Classified {len(detections)} items in {resnet_timer.elapsed:.2f}s")
        
        # Calculate summary
        fresh_count = sum(1 for d in detections if d['spoilage_status'] == 'fresh')
        spoiled_count = sum(1 for d in detections if d['spoilage_status'] == 'spoiled')
        
        summary = {
            'total_detections': len(detections),
            'fresh_count': fresh_count,
            'spoiled_count': spoiled_count,
            'fresh_percentage': (fresh_count / len(detections) * 100) if detections else 0,
            'spoiled_percentage': (spoiled_count / len(detections) * 100) if detections else 0
        }
        
        # Metadata
        processing_time = time.time() - start_time
        metadata = {
            'image_name': image_name,
            'image_shape': img_array.shape[:2],
            'processing_time': processing_time,
            'yolo_time': yolo_timer.elapsed,
            'resnet_time': resnet_timer.elapsed,
            'timestamp': get_timestamp()
        }
        
        # Build result
        result = {
            'detections': detections,
            'summary': summary,
            'metadata': metadata
        }
        
        # Save visualization
        if save_visualization or self.output_config.get('save_visualization', False):
            output_dir = output_dir or self.output_config.get('save_dir', 'results/pipeline_output')
            vis_path = self._save_visualization(img_array, detections, image_name, output_dir)
            result['visualization_path'] = str(vis_path)
            self.logger.info(f"Visualization saved: {vis_path}")
        
        # Save crops
        if save_crops or self.output_config.get('save_crops', False):
            output_dir = output_dir or self.output_config.get('save_dir', 'results/pipeline_output')
            self._save_crops(crops_list, detections, image_name, output_dir)
        
        # Include crops in result
        if return_crops:
            result['crops'] = crops_list
        
        self.logger.info(f"Pipeline complete: {fresh_count} fresh, {spoiled_count} spoiled (total time: {processing_time:.2f}s)")
        
        return result
    
    def _save_visualization(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        image_name: str,
        output_dir: Union[str, Path]
    ) -> Path:
        """Draw bounding boxes and labels on image and save."""
        output_dir = ensure_dir(output_dir)
        
        # Copy image for annotation
        img_vis = image.copy()
        
        # Get visualization settings
        vis_config = self.output_config.get('visualization', {})
        thickness = vis_config.get('bbox_thickness', 2)
        font_scale = vis_config.get('font_scale', 0.6)
        colors = vis_config.get('colors', {
            'fresh': [0, 255, 0],
            'spoiled': [0, 0, 255]
        })
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            status = det['spoilage_status']
            
            # Get color
            color = tuple(colors.get(status, [255, 255, 0]))  # Yellow for unknown
            
            # Draw bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)
            
            # Create label
            label = f"{det['object_class']}: {status}"
            conf_label = f"{det['spoilage_confidence']:.2%}"
            
            # Draw label background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(img_vis, (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(img_vis, label, (x1, y1 - baseline - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Draw confidence below box
            cv2.putText(img_vis, conf_label, (x1, y2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, 1)
        
        # Save annotated image
        output_path = output_dir / f"{image_name}_annotated.jpg"
        # Convert RGB to BGR for OpenCV
        img_vis_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_vis_bgr)
        
        return output_path
    
    def _save_crops(
        self,
        crops: List[np.ndarray],
        detections: List[Dict[str, Any]],
        image_name: str,
        output_dir: Union[str, Path]
    ):
        """Save individual crops."""
        crop_dir = ensure_dir(Path(output_dir) / 'crops' / image_name)
        
        for i, (crop, det) in enumerate(zip(crops, detections)):
            status = det['spoilage_status']
            obj_class = det['object_class']
            conf = det['spoilage_confidence']
            
            filename = f"crop_{i:03d}_{obj_class}_{status}_{conf:.2f}.jpg"
            crop_path = crop_dir / filename
            
            # Convert RGB to BGR for OpenCV
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(crop_path), crop_bgr)
        
        self.logger.info(f"Saved {len(crops)} crops to {crop_dir}")
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        save_visualization: bool = False,
        save_results: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images through the pipeline.
        
        Args:
            image_paths (list): List of image paths
            save_visualization (bool): Save annotated images
            save_results (bool): Save results to JSON
            output_dir (str or Path, optional): Output directory
        
        Returns:
            list: List of results for each image
        
        Example:
            >>> images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
            >>> results = pipeline.process_batch(images, save_visualization=True)
        """
        self.logger.info(f"Processing batch of {len(image_paths)} images")
        
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            self.logger.info(f"[{i}/{len(image_paths)}] Processing {image_path}")
            
            try:
                result = self.process_image(
                    image_path,
                    save_visualization=save_visualization,
                    output_dir=output_dir
                )
                results.append(result)
            
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'error': str(e),
                    'image_path': str(image_path)
                })
        
        # Calculate batch summary
        total_detections = sum(r.get('summary', {}).get('total_detections', 0) for r in results)
        total_fresh = sum(r.get('summary', {}).get('fresh_count', 0) for r in results)
        total_spoiled = sum(r.get('summary', {}).get('spoiled_count', 0) for r in results)
        
        batch_summary = {
            'total_images': len(image_paths),
            'successful_images': len([r for r in results if 'error' not in r]),
            'total_detections': total_detections,
            'total_fresh': total_fresh,
            'total_spoiled': total_spoiled
        }
        
        self.logger.info(f"Batch processing complete: {batch_summary}")
        
        # Save batch results
        if save_results:
            output_dir = output_dir or self.output_config.get('save_dir', 'results/pipeline_output')
            output_dir = ensure_dir(output_dir)
            
            batch_results = {
                'batch_summary': batch_summary,
                'individual_results': results
            }
            
            results_file = output_dir / f"batch_results_{get_timestamp()}.json"
            save_json(batch_results, results_file)
            self.logger.info(f"Batch results saved: {results_file}")
        
        return results


# ============================================================================
# Standalone Pipeline Function
# ============================================================================

def run_pipeline(
    image_path: Union[str, Path],
    config_path: Optional[str] = 'configs/pipeline_config.yaml',
    yolo_model: Optional[str] = None,
    resnet_model: Optional[str] = None,
    save_visualization: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the complete food spoilage detection pipeline on a single image.
    
    Args:
        image_path (str or Path): Path to input image
        config_path (str, optional): Path to pipeline config
        yolo_model (str, optional): Path to YOLO model (overrides config)
        resnet_model (str, optional): Path to ResNet model (overrides config)
        save_visualization (bool): Save annotated image
        output_dir (str, optional): Custom output directory
    
    Returns:
        dict: Pipeline results
    
    Example:
        >>> results = run_pipeline('food_image.jpg')
        >>> print(f"Found {results['summary']['total_detections']} items")
        >>> print(f"Fresh: {results['summary']['fresh_count']}")
        >>> print(f"Spoiled: {results['summary']['spoiled_count']}")
    """
    pipeline = FoodSpoilagePipeline(
        config_path=config_path,
        yolo_model_path=yolo_model,
        resnet_model_path=resnet_model
    )
    
    return pipeline.process_image(
        image_path,
        save_visualization=save_visualization,
        output_dir=output_dir
    )


# ============================================================================
# Main / CLI
# ============================================================================

def main():
    """Main function for CLI pipeline execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Food Spoilage Detection Pipeline')
    parser.add_argument('image', type=str, nargs='?', help='Path to input image')
    parser.add_argument('--config', type=str, default='configs/pipeline_config.yaml',
                       help='Path to pipeline config')
    parser.add_argument('--yolo-model', type=str, default=None,
                       help='Path to YOLO model')
    parser.add_argument('--resnet-model', type=str, default=None,
                       help='Path to ResNet model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory')
    parser.add_argument('--no-viz', action='store_true',
                       help='Do not save visualization')
    parser.add_argument('--save-crops', action='store_true',
                       help='Save individual crops')
    parser.add_argument('--batch', nargs='+',
                       help='Process multiple images')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = FoodSpoilagePipeline(
            config_path=args.config,
            yolo_model_path=args.yolo_model,
            resnet_model_path=args.resnet_model,
            device=args.device
        )
        
        if args.batch:
            # Batch processing
            results = pipeline.process_batch(
                args.batch,
                save_visualization=not args.no_viz,
                output_dir=args.output
            )
            
            print("\n" + "="*60)
            print("BATCH PROCESSING RESULTS")
            print("="*60)
            for i, result in enumerate(results, 1):
                if 'error' in result:
                    print(f"\n[{i}] {result['image_path']}: ERROR - {result['error']}")
                else:
                    summary = result['summary']
                    print(f"\n[{i}] {result['metadata']['image_name']}:")
                    print(f"  Total: {summary['total_detections']}, "
                          f"Fresh: {summary['fresh_count']}, "
                          f"Spoiled: {summary['spoiled_count']}")
        
        elif args.image:
            # Single image processing
            result = pipeline.process_image(
                args.image,
                save_visualization=not args.no_viz,
                save_crops=args.save_crops,
                output_dir=args.output
            )
            
            # Print results
            print("\n" + "="*60)
            print("PIPELINE RESULTS")
            print("="*60)
            print(f"\nImage: {result['metadata']['image_name']}")
            print(f"Processing time: {result['metadata']['processing_time']:.2f}s")
            print(f"\nSummary:")
            print(f"  Total detections: {result['summary']['total_detections']}")
            print(f"  Fresh:   {result['summary']['fresh_count']} ({result['summary']['fresh_percentage']:.1f}%)")
            print(f"  Spoiled: {result['summary']['spoiled_count']} ({result['summary']['spoiled_percentage']:.1f}%)")
            
            print(f"\nDetailed Results:")
            for det in result['detections']:
                print(f"  [{det['detection_id']}] {det['object_class']}: {det['spoilage_status']} "
                      f"(confidence: {det['spoilage_confidence']:.2%})")
            
            if 'visualization_path' in result:
                print(f"\nVisualization saved: {result['visualization_path']}")
        
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
