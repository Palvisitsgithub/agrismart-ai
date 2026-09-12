"""Evaluate a saved classifier checkpoint on a directory of class folders."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    transform = transforms.Compose([
        transforms.Resize((checkpoint.get("image_size", 224), checkpoint.get("image_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    unknown = sorted(set(dataset.classes) - set(classes))
    if unknown:
        raise ValueError(f"Dataset contains unsupported classes: {unknown}")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    truth, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            predictions.extend(logits.argmax(1).tolist())
            truth.extend(dataset.classes[label] for label in labels)
            predictions[-len(labels):] = [classes[index] for index in predictions[-len(labels):]]
    result = {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0),
        "images": len(truth),
        "classes": sorted(set(truth)),
        "classification_report": classification_report(truth, predictions, zero_division=0, output_dict=True),
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
