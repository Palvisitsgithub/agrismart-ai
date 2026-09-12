"""Download a Kaggle dataset without committing the raw files to Git."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True, help="Kaggle dataset slug, e.g. owner/dataset-name")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.slug,
        "--unzip",
        "-p",
        str(args.output_dir),
    ]
    print("Downloading dataset files to", args.output_dir)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
