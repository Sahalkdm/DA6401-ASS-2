"""Training entrypoint

Usage:
    python train.py --data_root ./pets_data --epochs 30
    python train.py --data_root ./pets_data --epochs 30 --use_wandb
    python train.py --data_root ./pets_data --dropout_p 0.0   # ablation: no dropout
    python train.py --data_root ./pets_data --dropout_p 0.2   # ablation: light dropout
"""

import argparse
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier


def parse_args():
    p = argparse.ArgumentParser(description="Train VGG11 on Oxford-IIIT Pet")
    p.add_argument("--data_root", type=str, default="./pets_data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--dropout_p", type=float, default=0.6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="da6401_assignment2")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def get_train_transform(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"image": images, "label": labels}


def progress_bar(current, total, prefix="", suffix="", width=30):
    filled = int(width * current / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    pct = 100 * current / max(total, 1)
    print(f"\r{prefix} |{bar}| {pct:5.1f}% {suffix}", end="", flush=True)
    if current >= total:
        print()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(1) == labels).sum().item()
        total += bs

        elapsed = time.time() - t0
        steps_done = step + 1
        eta = elapsed / steps_done * (len(loader) - steps_done)

        suffix = f"loss={running_loss / total:.4f} acc={correct / total:.3f} ETA {eta:.0f}s"

        progress_bar(steps_done, len(loader),
                     prefix=f"Epoch {epoch:02d}/{total_epochs} [train]",
                     suffix=suffix)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, split_name="val"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    t0 = time.time()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        eta = (time.time() - t0) / (step + 1) * (len(loader) - step - 1)

        progress_bar(step + 1, len(loader),
                     prefix=f"[{split_name}]",
                     suffix=f"ETA {eta:.0f}s")

    n = len(all_labels)
    val_loss = running_loss / n
    val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return val_loss, val_acc, val_f1


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")
        args.amp = False
        print("No CUDA found. Running on CPU.")

    use_amp = args.amp and device.type == "cuda"

    print(f"Device={device} AMP={use_amp} batch_size={args.batch_size}")

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project,
                   name=args.run_name or f"vgg11_drop{args.dropout_p}",
                   config=vars(args))

    train_tf = get_train_transform(args.image_size)
    val_tf = get_val_transform(args.image_size)

    full_ds = OxfordIIITPetDataset(root=args.data_root, split="trainval",
                                   transform=train_tf, download=args.download)

    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val

    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                   generator=torch.Generator().manual_seed(args.seed))

    val_ds.dataset.transform = val_tf

    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test",
                                  transform=val_tf)

    loader_kw = dict(batch_size=args.batch_size,
                     collate_fn=collate_fn,
                     pin_memory=(device.type == "cuda"),
                     num_workers=args.num_workers,
                     persistent_workers=(args.num_workers > 0))

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    print(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    model = VGG11Classifier(num_classes=37, in_channels=3,
                           dropout_p=args.dropout_p).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, args.epochs, use_amp
        )

        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device, use_amp, "val"
        )

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        print(f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} f1={val_f1:.3f} | "
              f"lr={lr_now:.2e}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "args": vars(args)
            }, Path(args.checkpoint_dir) / "best_classifier.pth")

    ckpt = torch.load(Path(args.checkpoint_dir) / "best_classifier.pth",
                      map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_f1 = evaluate(
        model, test_loader, criterion, device, use_amp, "test"
    )

    print(f"FINAL TEST loss={test_loss:.4f} acc={test_acc:.3f} f1={test_f1:.3f}")


if __name__ == "__main__":
    main()
