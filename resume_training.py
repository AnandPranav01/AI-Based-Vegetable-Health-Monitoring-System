"""Resume YOLO training from last checkpoint with optimized settings for speed"""
from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # Print GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load the last checkpoint from yolo_train2
    model = YOLO("runs/detect/yolo_train2/weights/last.pt")

    # Resume training with optimized settings for maximum speed
    # With increased paging size, we can use higher batch size and workers
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    results = model.train(
        resume=True,
        data="data/yolo_dataset/data.yaml",
        epochs=100,

        batch=8,
        imgsz=512,

        workers=0
        ,
        cache=False,

        device=0,
        amp=True,

        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,

        close_mosaic=10,
        patience=30,
        save_period=10,
        plots=False,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Training resumed and completed!")
    print("Best model saved to: runs/detect/yolo_train2/weights/best.pt")
    print("=" * 60)
