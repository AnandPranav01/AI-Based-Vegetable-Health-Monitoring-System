import os
import random
import shutil
from pathlib import Path

# ---------------- CONFIG ---------------- #
RAW_IMAGES_DIR = Path("data/raw/yolo/images")
RAW_LABELS_DIR = Path("data/raw/yolo/labels")

OUTPUT_BASE = Path("data/yolo_dataset")

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

RANDOM_SEED = 42
# ---------------------------------------- #

def main():
    random.seed(RANDOM_SEED)

    # Validate source directories
    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {RAW_IMAGES_DIR}")
    if not RAW_LABELS_DIR.exists():
        raise FileNotFoundError(f"Labels directory not found: {RAW_LABELS_DIR}")

    # Create output directories
    for split in ["train", "val", "test"]:
        (OUTPUT_BASE / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_BASE / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = [
        f.name for f in RAW_IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    if not image_files:
        raise ValueError(f"No images found in {RAW_IMAGES_DIR}")

    image_files.sort()
    random.shuffle(image_files)

    total_images = len(image_files)

    train_end = int(SPLIT_RATIO["train"] * total_images)
    val_end = train_end + int(SPLIT_RATIO["val"] * total_images)

    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    # Track statistics
    stats = {
        "train": {"copied": 0, "skipped": 0},
        "val": {"copied": 0, "skipped": 0},
        "test": {"copied": 0, "skipped": 0}
    }

    def copy_files(file_list, split_name):
        for img_file in file_list:
            label_file = Path(img_file).stem + ".txt"

            src_img = RAW_IMAGES_DIR / img_file
            src_lbl = RAW_LABELS_DIR / label_file

            dst_img = OUTPUT_BASE / "images" / split_name / img_file
            dst_lbl = OUTPUT_BASE / "labels" / split_name / label_file

            if not src_lbl.exists():
                print(f"[WARNING] Label missing for {img_file}, skipping.")
                stats[split_name]["skipped"] += 1
                continue

            try:
                shutil.copy(src_img, dst_img)
                shutil.copy(src_lbl, dst_lbl)
                stats[split_name]["copied"] += 1
            except Exception as e:
                print(f"[ERROR] Failed to copy {img_file}: {e}")
                stats[split_name]["skipped"] += 1

    for split_name, files in splits.items():
        copy_files(files, split_name)

    print("\n✅ YOLO dataset split completed successfully!")
    print(f"\nTotal images found: {total_images}")
    print(f"\nTrain: {stats['train']['copied']} copied, {stats['train']['skipped']} skipped")
    print(f"Val:   {stats['val']['copied']} copied, {stats['val']['skipped']} skipped")
    print(f"Test:  {stats['test']['copied']} copied, {stats['test']['skipped']} skipped")
    print(f"\nTotal copied: {sum(s['copied'] for s in stats.values())}")
    print(f"Total skipped: {sum(s['skipped'] for s in stats.values())}")

if __name__ == "__main__":
    main()
