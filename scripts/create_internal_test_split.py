"""Create train/validation/test folders from an existing train/val dataset."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_files(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        shutil.copy2(source, destination / source.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_ratio < 1:
        raise ValueError("test ratio must be between 0 and 1")
    rng = random.Random(args.seed)

    for class_dir in sorted(path for path in args.train_dir.iterdir() if path.is_dir()):
        images = [path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
        rng.shuffle(images)
        test_count = max(1, round(len(images) * args.test_ratio))
        copy_files(images[test_count:], args.output_dir / "train" / class_dir.name)
        copy_files(images[:test_count], args.output_dir / "test" / class_dir.name)

    for class_dir in sorted(path for path in args.val_dir.iterdir() if path.is_dir()):
        images = [path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
        copy_files(images, args.output_dir / "val" / class_dir.name)

    print(f"Prepared train/test split from {args.train_dir}")
    print(f"Copied validation split from {args.val_dir}")


if __name__ == "__main__":
    main()
