"""
Food Spoilage Detection System - Main Entry Point

This is the main entry point for the food spoilage detection system.
It provides a unified CLI interface for all system operations:

- Pipeline: Run complete detection and classification pipeline
- Train: Train ResNet classification model
- Evaluate: Evaluate trained models
- Detect: Run YOLO object detection only
- Classify: Run ResNet classification only

Usage:
    python main.py pipeline <image_path> [options]
    python main.py train [options]
    python main.py evaluate [options]
    python main.py detect <image_path> [options]
    python main.py classify <image_path> [options]
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pipeline.inference import FoodSpoilagePipeline, run_pipeline
from src.resnet.train_resnet import train_resnet, ResNetTrainer
from src.resnet.evaluate_resnet import evaluate_resnet, ResNetEvaluator
from src.resnet.predict_resnet import predict as resnet_predict
from src.yolo.predict_yolo import predict as yolo_predict


def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║       Food Spoilage Detection System                     ║
    ║       Two-Stage Detection & Classification               ║
    ║                                                          ║
    ║       YOLO (Detection) + ResNet (Classification)         ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def format_pipeline_results(result: dict):
    """Format and print pipeline results in a nice way."""
    print("\n" + "="*70)
    print("PIPELINE RESULTS")
    print("="*70)
    
    metadata = result.get('metadata', {})
    summary = result.get('summary', {})
    detections = result.get('detections', [])
    
    # Image info
    print(f"\n📷 Image: {metadata.get('image_name', 'N/A')}")
    print(f"⏱️  Processing Time: {metadata.get('processing_time', 0):.2f}s")
    print(f"   - YOLO Detection: {metadata.get('yolo_time', 0):.2f}s")
    print(f"   - ResNet Classification: {metadata.get('resnet_time', 0):.2f}s")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total Detections: {summary.get('total_detections', 0)}")
    print(f"   ✅ Fresh:   {summary.get('fresh_count', 0)} ({summary.get('fresh_percentage', 0):.1f}%)")
    print(f"   ❌ Spoiled: {summary.get('spoiled_count', 0)} ({summary.get('spoiled_percentage', 0):.1f}%)")
    
    # Detailed detections
    if detections:
        print(f"\n🔍 Detailed Results:")
        print(f"   {'#':<3} {'Object':<15} {'Status':<10} {'Confidence':<12} {'Combined':<10}")
        print("   " + "-"*55)
        
        for det in detections:
            det_id = det.get('detection_id', 0)
            obj_class = det.get('object_class', 'unknown')
            status = det.get('spoilage_status', 'unknown')
            conf = det.get('spoilage_confidence', 0)
            combined = det.get('combined_confidence', 0)
            
            # Use emoji for status
            status_emoji = "✅" if status == 'fresh' else "❌"
            
            print(f"   {det_id:<3} {obj_class:<15} {status_emoji} {status:<8} "
                  f"{conf:>6.1%}       {combined:>6.1%}")
    
    # Output files
    if 'visualization_path' in result:
        print(f"\n💾 Visualization saved: {result['visualization_path']}")
    
    print("\n" + "="*70 + "\n")


def run_pipeline_mode(args):
    """Run the complete detection and classification pipeline."""
    print_banner()
    print("\n🚀 Running Two-Stage Pipeline (YOLO + ResNet)...\n")
    
    try:
        if args.batch:
            # Batch processing
            pipeline = FoodSpoilagePipeline(
                config_path=args.config,
                device=args.device
            )
            
            results = pipeline.process_batch(
                args.batch,
                save_visualization=args.save_viz,
                output_dir=args.output
            )
            
            print(f"\n✅ Processed {len(results)} images")
            
            # Print batch summary
            total_det = sum(r.get('summary', {}).get('total_detections', 0) for r in results)
            total_fresh = sum(r.get('summary', {}).get('fresh_count', 0) for r in results)
            total_spoiled = sum(r.get('summary', {}).get('spoiled_count', 0) for r in results)
            
            print(f"\n📊 Batch Summary:")
            print(f"   Total Detections: {total_det}")
            print(f"   Fresh: {total_fresh}")
            print(f"   Spoiled: {total_spoiled}")
        
        else:
            # Single image
            result = run_pipeline(
                args.image,
                config_path=args.config,
                save_visualization=args.save_viz,
                output_dir=args.output
            )
            
            format_pipeline_results(result)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_train_mode(args):
    """Train ResNet model."""
    print_banner()
    print("\n🎓 Training ResNet Model...\n")
    
    try:
        train_resnet(
            config_path=args.config,
            resume_from=args.resume
        )
        
        print("\n✅ Training completed successfully!")
        return 0
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_evaluate_mode(args):
    """Evaluate trained model."""
    print_banner()
    print("\n📈 Evaluating Model...\n")
    
    try:
        metrics = evaluate_resnet(
            model_path=args.model,
            config_path=args.config,
            dataset_split=args.split,
            save_results=not args.no_save,
            output_dir=args.output,
            device=args.device
        )
        
        print(f"\n✅ Evaluation completed!")
        print(f"   Accuracy: {metrics['overall']['accuracy']:.2%}")
        print(f"   F1-Score: {metrics['overall']['f1_macro']:.4f}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_detect_mode(args):
    """Run YOLO detection only."""
    print_banner()
    print("\n🎯 Running YOLO Detection...\n")
    
    try:
        detections = yolo_predict(
            args.image,
            model_path=args.model or 'models/yolo_best.pt',
            save_results=args.save_viz,
            confidence=args.conf,
            device=args.device
        )
        
        print(f"✅ Detected {len(detections)} objects:\n")
        
        for i, det in enumerate(detections, 1):
            print(f"   {i}. {det['class_name']}: {det['confidence']:.2%} "
                  f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
                  f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Detection failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_classify_mode(args):
    """Run ResNet classification only."""
    print_banner()
    print("\n🏷️  Running ResNet Classification...\n")
    
    try:
        result = resnet_predict(
            args.image,
            model_path=args.model or 'models/resnet_spoilage.pt',
            device=args.device,
            return_confidence=True
        )
        
        print(f"✅ Classification Result:\n")
        print(f"   Class: {result['class']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"\n   Probabilities:")
        for class_name, prob in result['probabilities'].items():
            emoji = "✅" if class_name == 'fresh' else "❌"
            print(f"      {emoji} {class_name}: {prob:.2%}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Classification failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Food Spoilage Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py pipeline sample.jpg
  python main.py pipeline sample.jpg --save-viz --output results
  
  # Batch processing
  python main.py pipeline --batch img1.jpg img2.jpg img3.jpg
  
  # Train ResNet model
  python main.py train --config configs/resnet_config.yaml
  python main.py train --resume models/checkpoint.pt
  
  # Evaluate model
  python main.py evaluate --model models/resnet_spoilage.pt
  python main.py evaluate --split test
  
  # YOLO detection only
  python main.py detect sample.jpg --conf 0.5
  
  # ResNet classification only
  python main.py classify crop.jpg --model models/resnet_spoilage.pt
        """
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # ========================================================================
    # Pipeline mode
    # ========================================================================
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('image', nargs='?', type=str, 
                                help='Input image path')
    pipeline_parser.add_argument('--batch', nargs='+',
                                help='Process multiple images')
    pipeline_parser.add_argument('--config', type=str, 
                                default='configs/pipeline_config.yaml',
                                help='Pipeline configuration file')
    pipeline_parser.add_argument('--output', type=str,
                                help='Output directory')
    pipeline_parser.add_argument('--save-viz', action='store_true',
                                help='Save visualization')
    pipeline_parser.add_argument('--device', type=str,
                                help='Device (cuda/cpu)')
    pipeline_parser.add_argument('--verbose', action='store_true',
                                help='Verbose output')
    
    # ========================================================================
    # Train mode
    # ========================================================================
    train_parser = subparsers.add_parser('train', help='Train ResNet model')
    train_parser.add_argument('--config', type=str, 
                             default='configs/resnet_config.yaml',
                             help='Training configuration file')
    train_parser.add_argument('--resume', type=str,
                             help='Resume from checkpoint')
    train_parser.add_argument('--verbose', action='store_true',
                             help='Verbose output')
    
    # ========================================================================
    # Evaluate mode
    # ========================================================================
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', type=str, 
                            default='models/resnet_spoilage.pt',
                            help='Model checkpoint path')
    eval_parser.add_argument('--config', type=str,
                            help='Configuration file')
    eval_parser.add_argument('--split', type=str, default='val',
                            choices=['val', 'test'],
                            help='Dataset split')
    eval_parser.add_argument('--output', type=str, default='results',
                            help='Output directory')
    eval_parser.add_argument('--device', type=str,
                            help='Device (cuda/cpu)')
    eval_parser.add_argument('--no-save', action='store_true',
                            help='Do not save results')
    eval_parser.add_argument('--verbose', action='store_true',
                            help='Verbose output')
    
    # ========================================================================
    # Detect mode (YOLO only)
    # ========================================================================
    detect_parser = subparsers.add_parser('detect', help='YOLO detection only')
    detect_parser.add_argument('image', type=str, help='Input image path')
    detect_parser.add_argument('--model', type=str,
                              help='YOLO model path')
    detect_parser.add_argument('--conf', type=float, default=0.25,
                              help='Confidence threshold')
    detect_parser.add_argument('--save-viz', action='store_true',
                              help='Save visualization')
    detect_parser.add_argument('--device', type=str,
                              help='Device (cuda/cpu)')
    detect_parser.add_argument('--verbose', action='store_true',
                              help='Verbose output')
    
    # ========================================================================
    # Classify mode (ResNet only)
    # ========================================================================
    classify_parser = subparsers.add_parser('classify', help='ResNet classification only')
    classify_parser.add_argument('image', type=str, help='Input image path')
    classify_parser.add_argument('--model', type=str,
                                help='ResNet model path')
    classify_parser.add_argument('--device', type=str,
                                help='Device (cuda/cpu)')
    classify_parser.add_argument('--verbose', action='store_true',
                                help='Verbose output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no mode specified, show help
    if not args.mode:
        parser.print_help()
        return 0
    
    # Route to appropriate mode
    if args.mode == 'pipeline':
        if not args.image and not args.batch:
            pipeline_parser.print_help()
            return 1
        return run_pipeline_mode(args)
    
    elif args.mode == 'train':
        return run_train_mode(args)
    
    elif args.mode == 'evaluate':
        return run_evaluate_mode(args)
    
    elif args.mode == 'detect':
        return run_detect_mode(args)
    
    elif args.mode == 'classify':
        return run_classify_mode(args)
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
