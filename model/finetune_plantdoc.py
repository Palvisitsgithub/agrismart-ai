"""Fine-tune the PlantVillage checkpoint on mapped PlantDoc training crops."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

class RemappedDataset(Dataset):
    def __init__(self, base, class_to_index):
        self.base = base
        self.indices = [i for i, label in enumerate(base.targets) if base.classes[label] in class_to_index]
        self.targets = [class_to_index[base.classes[base.targets[i]]] for i in self.indices]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        image, _ = self.base[self.indices[index]]
        return image, self.targets[index]

def evaluate(model, loader, device):
    model.eval(); truth, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    return f1_score(truth, predictions, average="macro", zero_division=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]
    class_to_index = {name: index for index, name in enumerate(classes)}
    image_size = checkpoint.get("image_size", 224)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])
    train_base = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_base = datasets.ImageFolder(args.val_dir, transform=eval_transform)
    train_set = RemappedDataset(train_base, class_to_index)
    val_set = RemappedDataset(val_base, class_to_index)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"finetune {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward(); optimizer.step()
        score = evaluate(model, val_loader, device)
        print(f"epoch={epoch} plantdoc_val_macro_f1={score:.4f}", flush=True)
        if score > best_f1:
            best_f1 = score
            torch.save({"state_dict": model.state_dict(), "classes": classes, "image_size": image_size}, args.output)
    print(f"Saved best fine-tuned checkpoint to {args.output}")

if __name__ == "__main__":
    main()
