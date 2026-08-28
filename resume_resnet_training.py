"""
Resume ResNet training from the latest checkpoint.

This script automatically finds the most recent checkpoint in the models/
directory and resumes training from where it left off, restoring epoch count,
optimizer state, scheduler state, training history, and best metrics.

Usage:
    python resume_resnet_training.py
    python resume_resnet_training.py --checkpoint models/checkpoint_epoch_20.pt
    python resume_resnet_training.py --checkpoint models/checkpoint_interrupted.pt
    python resume_resnet_training.py --config configs/resnet_config.yaml --device cpu
"""

import sys
import argparse
from pathlib import Path
import torch

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

from resnet.train_resnet import ResNetTrainer
from pipeline.utils import load_config


def find_latest_checkpoint(checkpoint_dir: str = 'models') -> Path:
    """
    Automatically find the most recent checkpoint file.

    Priority order:
      1. checkpoint_interrupted.pt  (saved on keyboard interrupt – resume immediately)
      2. checkpoint_epoch_N.pt      (latest epoch, by epoch number)

    Args:
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        Path to the latest checkpoint.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    search_dir = Path(checkpoint_dir)

    if not search_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {search_dir}")

    # Priority 1: interrupted checkpoint
    interrupted = search_dir / 'checkpoint_interrupted.pt'
    if interrupted.exists():
        print(f"  Found interrupted checkpoint: {interrupted}")
        return interrupted

    # Priority 2: highest-numbered epoch checkpoint
    epoch_checkpoints = sorted(
        search_dir.glob('checkpoint_epoch_*.pt'),
        key=lambda p: int(p.stem.split('_')[-1])   # sort by epoch number
    )

    if epoch_checkpoints:
        latest = epoch_checkpoints[-1]
        print(f"  Found {len(epoch_checkpoints)} epoch checkpoint(s). Using latest: {latest}")
        return latest

    raise FileNotFoundError(
        f"No checkpoints found in '{search_dir}'.\n"
        "  - Run training first with:  python main.py train\n"
        "  - Or specify a path with:   --checkpoint <path>"
    )


def print_checkpoint_info(checkpoint_path: Path, device: torch.device):
    """Print a summary of what's inside the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        print("  [!] Checkpoint is a bare state dict — no training state to restore.")
        return

    epoch       = checkpoint.get('epoch', 'unknown')
    best_acc    = checkpoint.get('best_val_accuracy', 0.0)
    best_loss   = checkpoint.get('best_val_loss', float('inf'))
    history     = checkpoint.get('history', {})
    train_losses = history.get('train_loss', [])

    print(f"  Stopped at epoch    : {epoch + 1 if isinstance(epoch, int) else epoch}")
    print(f"  Best val accuracy   : {best_acc:.2f}%")
    print(f"  Best val loss       : {best_loss:.4f}")
    print(f"  Epochs recorded     : {len(train_losses)}")


def main():
    parser = argparse.ArgumentParser(
        description='Resume ResNet training from a checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resume_resnet_training.py
  python resume_resnet_training.py --checkpoint models/checkpoint_epoch_15.pt
  python resume_resnet_training.py --device cpu
        """
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to checkpoint file. Auto-detected if not specified.'
    )
    parser.add_argument(
        '--config', type=str, default='configs/resnet_config.yaml',
        help='Path to ResNet config file (default: configs/resnet_config.yaml)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device override: cuda or cpu (uses config value if not specified)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='models',
        help='Directory to search for checkpoints (default: models)'
    )

    args = parser.parse_args()

    # ── GPU info ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("ResNet Training — Resume")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version  : {torch.version.cuda}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU memory    : {mem_gb:.2f} GB")
    else:
        print("GPU           : Not available — using CPU")

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    print("\nLocating checkpoint...")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            return 1
        print(f"  Using specified checkpoint: {checkpoint_path}")
    else:
        try:
            checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            return 1

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"\n[ERROR] Config file not found: {config_path}")
        return 1

    config = load_config(str(config_path))

    # Override device if requested
    if args.device:
        config['device'] = args.device

    device_name = config.get('device', 'cuda')
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device  : {device}")

    # ── Show checkpoint summary ───────────────────────────────────────────────
    print("\nCheckpoint info:")
    print_checkpoint_info(checkpoint_path, device)

    # ── Resume training ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Resuming training...")
    print("=" * 60 + "\n")

    try:
        trainer = ResNetTrainer(
            config_dict=config,
            resume_from=str(checkpoint_path)
        )
        trainer.train()

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
        print("       A checkpoint has been saved as 'checkpoint_interrupted.pt'.")
        print("       Run this script again to resume.")
        return 0

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model saved to : models/resnet_spoilage.pt")
    print(f"Metrics saved to    : runs/resnet/resnet_train/training_metrics.json")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
