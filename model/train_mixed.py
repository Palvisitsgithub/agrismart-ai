"""Train a balanced mixed PlantVillage + PlantDoc classifier."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm

class MappedFolder(Dataset):
    def __init__(self, folder, class_to_index):
        self.folder = folder
        self.class_to_index = class_to_index
        self.indices = [i for i, label in enumerate(folder.targets) if folder.classes[label] in class_to_index]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, index):
        source_index = self.indices[index]
        image, label = self.folder[source_index]
        return image, self.class_to_index[self.folder.classes[label]]

def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval(); truth, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    return {"accuracy": accuracy_score(truth, predictions), "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plantvillage-dir", type=Path, required=True)
    parser.add_argument("--plantdoc-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint["classes"]; class_to_index = {name: i for i, name in enumerate(classes)}
    size = checkpoint.get("image_size", 224)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), normalize])
    pv_train = MappedFolder(datasets.ImageFolder(args.plantvillage_dir / "train", transform=train_transform), class_to_index)
    pd_train = MappedFolder(datasets.ImageFolder(args.plantdoc_dir / "train", transform=train_transform), class_to_index)
    pv_val = MappedFolder(datasets.ImageFolder(args.plantvillage_dir / "val", transform=eval_transform), class_to_index)
    pd_val = MappedFolder(datasets.ImageFolder(args.plantdoc_dir / "val", transform=eval_transform), class_to_index)
    mixed_loader = DataLoader(ConcatDataset([pv_train, pd_train]), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    pv_loader = DataLoader(pv_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    pd_loader = DataLoader(pd_val, batch_size=args.batch_size, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=None); model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes)); model.load_state_dict(checkpoint["state_dict"]); model.to(device)
    for parameter in model.features.parameters(): parameter.requires_grad = False
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(); best_score = -1.0; args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in tqdm(mixed_loader, desc=f"mixed {epoch}/{args.epochs}"):
            optimizer.zero_grad(); loss = criterion(model(images.to(device)), labels.to(device)); loss.backward(); optimizer.step()
        pv_score = evaluate(model, pv_loader, device); pd_score = evaluate(model, pd_loader, device); combined = (pv_score["macro_f1"] + pd_score["macro_f1"]) / 2
        print(f"epoch={epoch} plantvillage={pv_score} plantdoc={pd_score}", flush=True)
        if combined > best_score:
            best_score = combined; torch.save({"state_dict": model.state_dict(), "classes": classes, "image_size": size}, args.output)
    print(f"Saved mixed checkpoint to {args.output}")

if __name__ == "__main__": main()
