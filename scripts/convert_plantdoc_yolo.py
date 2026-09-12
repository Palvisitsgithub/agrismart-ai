"""Crop selected PlantDoc YOLO boxes into PlantVillage-compatible folders."""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image

CLASS_MAP = {
    2: "Corn_(maize)___Northern_Leaf_Blight", 3: "Apple___Cedar_apple_rust",
    4: "Potato___Late_blight", 6: "Corn_(maize)___Common_rust_",
    7: "Tomato___Late_blight", 8: "Tomato___Leaf_Mold", 9: "Potato___Early_blight",
    11: "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 13: "Tomato___Tomato_mosaic_virus",
    15: "Tomato___Bacterial_spot", 16: "Squash___Powdery_mildew",
    18: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 19: "Tomato___Early_blight",
    20: "Apple___Apple_scab", 21: "Tomato___Septoria_leaf_spot",
    26: "Grape___Black_rot", 28: "Tomato___Spider_mites Two-spotted_spider_mite",
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    written = 0
    for split in ("train", "val"):
        image_dir = args.dataset_dir / "images" / split
        label_dir = args.dataset_dir / "labels" / split
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            with Image.open(image_path).convert("RGB") as image:
                width, height = image.size
                for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                    values = [float(value) for value in line.split()]
                    if len(values) != 5 or int(values[0]) not in CLASS_MAP:
                        continue
                    class_id, xc, yc, bw, bh = values
                    left = max(0, int((xc - bw / 2) * width))
                    top = max(0, int((yc - bh / 2) * height))
                    right = min(width, int((xc + bw / 2) * width))
                    bottom = min(height, int((yc + bh / 2) * height))
                    if right <= left or bottom <= top:
                        continue
                    destination = args.output_dir / split / CLASS_MAP[int(class_id)]
                    destination.mkdir(parents=True, exist_ok=True)
                    image.crop((left, top, right, bottom)).save(destination / f"{image_path.stem}_{number}.jpg", quality=95)
                    written += 1
    print(f"Wrote {written} mapped PlantDoc crops to {args.output_dir}")

if __name__ == "__main__":
    main()
