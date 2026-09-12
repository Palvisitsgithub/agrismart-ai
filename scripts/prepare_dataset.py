"""Create reproducible train/validation/test folders from class directories."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def valid_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1:
        raise ValueError("Validation and test ratios must be between 0 and 1")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than 1")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.input_dir}")

    rng = random.Random(args.seed)
    class_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError("No class directories found in the input directory")

    copied = 0
    for class_dir in class_dirs:
        images = [path for path in class_dir.rglob("*") if valid_image(path)]
        rng.shuffle(images)
        n_test = round(len(images) * args.test_ratio)
        n_val = round(len(images) * args.val_ratio)
        splits = {
            "test": images[:n_test],
            "val": images[n_test : n_test + n_val],
            "train": images[n_test + n_val :],
        }
        for split, paths in splits.items():
            destination = args.output_dir / split / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for source in paths:
                target = destination / source.name
                if target.exists():
                    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
                    target = destination / f"{source.stem}_{digest}{source.suffix}"
                shutil.copy2(source, target)
                copied += 1

        print(f"{class_dir.name}: {len(images)} images")

    print(f"Prepared {copied} images in {args.output_dir}")


if __name__ == "__main__":
    main()
