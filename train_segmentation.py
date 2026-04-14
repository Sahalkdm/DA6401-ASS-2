import argparse
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.segmentation import VGG11UNet
from losses.dice_loss import DiceLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./pets_data")
    p.add_argument("--pretrained_clf", type=str, default=None,
                   help="Path to Task-1 classifier checkpoint (encoder weights).")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze the entire encoder.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout_p", type=float, default=0.3)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="da6401_assignment2")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download", action="store_true")
    p.add_argument("--warmup_epochs", type=int, default=3)
    return p.parse_args()


def get_train_transform(sz):
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(sz):
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def collate_fn_seg(batch):
    valid = [b for b in batch if b["mask"] is not None]
    if not valid:
        return None

    images = torch.stack([b["image"] for b in valid])

    masks = []
    for b in valid:
        m = b["mask"]
        if isinstance(m, torch.Tensor):
            m = m.long()
        else:
            m = torch.from_numpy(np.array(m)).long()

        m = (m - 1).clamp(0, 2)
        masks.append(m)

    masks = torch.stack(masks)
    return {"image": images, "mask": masks}


@torch.no_grad()
def compute_dice_score(logits, targets, num_classes=3, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    dice_scores = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        tgt_c = (targets == c).float()
        inter = (pred_c * tgt_c).sum()
        union = pred_c.sum() + tgt_c.sum()
        dice_scores.append(((2.0 * inter + smooth) / (union + smooth)).item())

    return float(np.mean(dice_scores))


@torch.no_grad()
def compute_pixel_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def pbar(cur, tot, prefix="", suffix="", w=28):
    filled = int(w * cur / max(tot, 1))
    print(f"\r{prefix} |{'#' * filled + '-' * (w - filled)}| {100 * cur / max(tot, 1):5.1f}% {suffix}", end="", flush=True)
    if cur >= tot:
        print()


def train_one_epoch(model, loader, ce_crit, dice_crit, optimizer, scaler, device, epoch, total_epochs, use_amp):
    model.train()
    total_loss = 0
    dice_sum = 0
    acc_sum = 0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            ce_loss = ce_crit(logits, masks)
            dice_loss = dice_crit(logits, masks)
            loss = ce_loss + 0.5 * dice_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        dice_sum += compute_dice_score(logits.detach(), masks)
        acc_sum += compute_pixel_accuracy(logits.detach(), masks)
        n_batches += 1

        done = step + 1
        eta = (time.time() - t0) / done * (len(loader) - done)

        pbar(done, len(loader),
             prefix=f"Epoch {epoch:02d}/{total_epochs} [train]",
             suffix=f"loss={total_loss / n_batches:.4f} dice={dice_sum / n_batches:.3f} ETA {eta:.0f}s")

    n = max(n_batches, 1)
    return total_loss / n, dice_sum / n, acc_sum / n


@torch.no_grad()
def evaluate(model, loader, ce_crit, dice_crit, device, use_amp, tag="val"):
    model.eval()
    total_loss = 0
    dice_sum = 0
    acc_sum = 0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            ce_loss = ce_crit(logits, masks)
            dice_loss = dice_crit(logits, masks)
            loss = ce_loss + 0.5 * dice_loss

        total_loss += loss.item()
        dice_sum += compute_dice_score(logits, masks)
        acc_sum += compute_pixel_accuracy(logits, masks)
        n_batches += 1

        done = step + 1
        eta = (time.time() - t0) / done * (len(loader) - done)

        pbar(done, len(loader), prefix=f"[{tag}]", suffix=f"ETA {eta:.0f}s")

    if n_batches == 0:
        print(f"No mask annotations found in {tag} split.")
        return 0.0, 0.0, 0.0

    return total_loss / n_batches, dice_sum / n_batches, acc_sum / n_batches


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        use_amp = True
        default_bs = 16
    else:
        device = torch.device("cpu")
        use_amp = False
        default_bs = 4
        print("No GPU found. Run check_gpu.py")

    batch_size = args.batch_size or default_bs
    sz = args.image_size

    print(f"Device={device} image_size={sz} batch={batch_size} AMP={use_amp}")

    if args.use_wandb:
        import wandb
        run_name = args.run_name or ("seg_frozen" if args.freeze_encoder else "seg_finetune")
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    full_ds = OxfordIIITPetDataset(root=args.data_root, split="trainval",
                                   transform=get_train_transform(sz), download=args.download)

    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val

    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                   generator=torch.Generator().manual_seed(args.seed))

    val_ds.dataset.transform = get_val_transform(sz)

    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test",
                                  transform=get_val_transform(sz))

    kw = dict(collate_fn=collate_fn_seg,
              num_workers=args.num_workers,
              pin_memory=(device.type == "cuda"),
              persistent_workers=(args.num_workers > 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    print(f"Train={n_train} Val={n_val} Test={len(test_ds)}")

    model = VGG11UNet(num_classes=3, in_channels=3,
                      dropout_p=args.dropout_p,
                      pretrained_clf=args.pretrained_clf,
                      freeze_encoder=args.freeze_encoder).to(device)

    class_weights = torch.tensor([1.0, 0.8, 3.0], device=device)
    class_weights = class_weights / class_weights.sum() * 3.0

    ce_crit = nn.CrossEntropyLoss(weight=class_weights)
    dice_crit = DiceLoss(num_classes=3, smooth=1.0)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = list(model.d5.parameters()) + list(model.d4.parameters()) + \
                 list(model.d3.parameters()) + list(model.d2.parameters()) + \
                 list(model.d1.parameters()) + list(model.output_conv.parameters())

    param_groups = [{"params": dec_params, "lr": args.lr}]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": args.lr / 10})

    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice, tr_acc = train_one_epoch(
            model, train_loader, ce_crit, dice_crit,
            optimizer, scaler, device, epoch, args.epochs, use_amp)

        vl_loss, vl_dice, vl_acc = evaluate(
            model, val_loader, ce_crit, dice_crit, device, use_amp, "val")

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"loss={tr_loss:.4f} dice={tr_dice:.3f} acc={tr_acc:.3f} | "
              f"val_loss={vl_loss:.4f} dice={vl_dice:.3f} acc={vl_acc:.3f} | "
              f"lr={lr_now:.2e}")

        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_dice": vl_dice,
                        "args": vars(args)},
                       Path(args.checkpoint_dir) / "best_segmentation.pth")

    ckpt = torch.load(Path(args.checkpoint_dir) / "best_segmentation.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    ts_loss, ts_dice, ts_acc = evaluate(
        model, test_loader, ce_crit, dice_crit, device, use_amp, "test")

    print(f"Test Dice={ts_dice:.4f} Test Acc={ts_acc:.4f} Test Loss={ts_loss:.4f}")


if __name__ == "__main__":
    main()
