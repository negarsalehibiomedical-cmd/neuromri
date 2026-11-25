"""Minimal research training scaffold. Dataset must be patient-level split."""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.model import ResearchCNN

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="backend/models/research_model.pt")
    args = p.parse_args()

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    root = Path(args.data)
    train = datasets.ImageFolder(root/"train", transform=tfm)
    val = datasets.ImageFolder(root/"val", transform=tfm)
    if train.classes != ["alzheimer_pattern", "control"] and set(train.classes) != {"control","alzheimer_pattern"}:
        raise ValueError("Class folders must be named control and alzheimer_pattern.")
    if set(val.classes) != set(train.classes):
        raise ValueError("Train/val class folders must match.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResearchCNN(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train(); total = 0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad()
            loss=loss_fn(model(x),y); loss.backward(); optimizer.step()
            total += loss.item()
        model.eval(); vloss=0
        with torch.no_grad():
            for x,y in val_loader:
                vloss += loss_fn(model(x.to(device)),y.to(device)).item()
        vloss /= max(1,len(val_loader))
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={total/max(1,len(train_loader)):.4f} val_loss={vloss:.4f}")
        if vloss < best:
            best=vloss
            out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
            torch.save(model.state_dict(),out)
    print("Saved best validation-loss checkpoint. Evaluate once on a locked test set.")

if __name__ == "__main__":
    main()
