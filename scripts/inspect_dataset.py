"""Print image counts by class for a directory-based image dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    classes = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not classes:
        raise SystemExit(f"No class directories found in {args.input_dir}")
    total = 0
    for class_dir in classes:
        count = sum(1 for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        print(f"{class_dir.name}\t{count}")
        total += count
    print(f"TOTAL\t{total}")


if __name__ == "__main__":
    main()
