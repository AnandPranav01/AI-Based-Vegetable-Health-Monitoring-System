"""
Setup script to create required model files for the Food Spoilage Detection pipeline.

- Downloads YOLOv8n pretrained model via ultralytics (auto-download)
- Creates a ResNet50 checkpoint with ImageNet pretrained backbone + random 2-class head

NOTE: The ResNet model uses pretrained ImageNet features but has NOT been
fine-tuned on food spoilage data. Predictions will be plausible but not
production-accurate until trained on a proper food freshness dataset.
"""

import torch
import shutil
from pathlib import Path

def setup_models():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # ---- 1. YOLO model ----
    print("="*60)
    print("Setting up YOLO model...")
    print("="*60)
    
    yolo_path = models_dir / "yolo_best.pt"
    if not yolo_path.exists():
        from ultralytics import YOLO
        # Download pretrained YOLOv8n (auto-downloads from ultralytics hub)
        model = YOLO("yolov8n.pt")
        # Copy the downloaded model to our expected path
        # ultralytics saves it to the current directory or cache
        downloaded = Path("yolov8n.pt")
        if downloaded.exists():
            shutil.copy(str(downloaded), str(yolo_path))
            print(f"  YOLO model saved to: {yolo_path}")
        else:
            # It may be in the ultralytics cache, just save via export
            model.save(str(yolo_path))
            print(f"  YOLO model saved to: {yolo_path}")
    else:
        print(f"  YOLO model already exists: {yolo_path}")

    # ---- 2. ResNet model ----
    print("\n" + "="*60)
    print("Setting up ResNet model...")
    print("="*60)
    
    resnet_path = models_dir / "resnet_spoilage.pt"
    if not resnet_path.exists():
        import sys
        sys.path.insert(0, "src")
        from resnet.model import build_resnet
        
        # Build ResNet50 with pretrained ImageNet backbone
        # The final layer is randomly initialized for 2 classes (fresh/spoiled)
        model = build_resnet(
            architecture='resnet50',
            num_classes=2,
            pretrained=True,  # Uses ImageNet pretrained weights for feature extraction
            dropout=0.5
        )
        
        # Save as checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'architecture': 'resnet50',
            'num_classes': 2,
            'dropout': 0.5,
            'note': 'ImageNet pretrained backbone, untrained classifier head'
        }
        torch.save(checkpoint, str(resnet_path))
        print(f"  ResNet model saved to: {resnet_path}")
    else:
        print(f"  ResNet model already exists: {resnet_path}")

    print("\n" + "="*60)
    print("Model setup complete!")
    print("="*60)
    print(f"\n  YOLO:   {yolo_path} ({'EXISTS' if yolo_path.exists() else 'MISSING'})")
    print(f"  ResNet: {resnet_path} ({'EXISTS' if resnet_path.exists() else 'MISSING'})")
    print("\nYou can now start the backend with:")
    print('  python -c "import uvicorn; uvicorn.run(\'deployment.app:app\', host=\'0.0.0.0\', port=8000)"')

if __name__ == "__main__":
    setup_models()
