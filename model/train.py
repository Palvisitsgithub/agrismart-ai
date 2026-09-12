"""Train an EfficientNet-B0 transfer-learning classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/efficientnet_b0"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_loaders(data_dir: Path, image_size: int, batch_size: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    train_set = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_set = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_set = datasets.ImageFolder(data_dir / "test", transform=eval_transform)
    kwargs = {"batch_size": batch_size, "num_workers": 0}
    return train_set, val_set, test_set, DataLoader(train_set, shuffle=True, **kwargs), DataLoader(val_set, shuffle=False, **kwargs), DataLoader(test_set, shuffle=False, **kwargs)


def evaluate(model, loader, device):
    model.eval()
    truth, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    return {
        "accuracy": accuracy_score(truth, predictions),
        "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set, train_loader, val_loader, test_loader = make_loaders(args.data_dir, args.image_size, args.batch_size)

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_set.classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} val_accuracy={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "classes": train_set.classes, "image_size": args.image_size}, args.output_dir / "best.pt")

    final_metrics = {"validation": evaluate(model, val_loader, device), "test": evaluate(model, test_loader, device), "classes": test_set.classes, "device": str(device)}
    (args.output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
