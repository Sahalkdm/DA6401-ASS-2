"""Training script: Localization """
"""Training script: Localization"""

import argparse
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.localization import VGG11Localizer
from losses.iou_loss import IoULoss, GIoULoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./pets_data")
    p.add_argument("--pretrained_clf", type=str, default=None)
    p.add_argument("--freeze_blocks", type=str, default="1-3", choices=["none", "1-3", "all"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4, help="LR for regression head. Encoder lr = lr/10.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout_p", type=float, default=0.5)
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


def get_transform(sz, train=True):
    ops = [A.Resize(sz, sz)]
    if train:
        ops += [A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5)]
    ops += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    return A.Compose(ops)


def collate_fn_bbox(batch):
    valid = [b for b in batch if b["bbox"] is not None]
    if not valid:
        return None
    return {
        "image": torch.stack([b["image"] for b in valid]),
        "bbox": torch.tensor([b["bbox"] for b in valid], dtype=torch.float32),
    }


@torch.no_grad()
def mean_iou(pred, tgt, eps=1e-6):
    def corners(b):
        return (
            b[:, 0] - b[:, 2] / 2,
            b[:, 1] - b[:, 3] / 2,
            b[:, 0] + b[:, 2] / 2,
            b[:, 1] + b[:, 3] / 2,
        )

    px1, py1, px2, py2 = corners(pred.clamp(0, 1))
    tx1, ty1, tx2, ty2 = corners(tgt)

    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)

    inter = iw * ih
    union = pred[:, 2].clamp(0) * pred[:, 3].clamp(0) + tgt[:, 2].clamp(0) * tgt[:, 3].clamp(0) - inter

    return (inter / (union + eps)).mean().item()


def pbar(cur, tot, prefix="", suffix="", w=28):
    filled = int(w * cur / max(tot, 1))
    print(f"\r{prefix} |{'#' * filled + '-' * (w - filled)}| {100 * cur / max(tot, 1):5.1f}% {suffix}", end="", flush=True)
    if cur >= tot:
        print()


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(model, loader, giou_crit, l1_crit, optimizer, scaler, device, epoch, total_epochs, use_amp):
    model.train()
    total_loss = 0
    iou_sum = 0
    n = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        images = batch["image"].to(device, non_blocking=True)
        tgt_norm = batch["bbox"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred_raw = model.forward_normalised(images)
            giou_loss = giou_crit(pred_raw, tgt_norm)
            l1_loss = l1_crit(pred_raw.clamp(0, 1), tgt_norm)
            loss = giou_loss + 0.5 * l1_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += loss.item() * bs
        iou_sum += mean_iou(pred_raw.detach(), tgt_norm) * bs
        n += bs

        done = step + 1
        eta = (time.time() - t0) / done * (len(loader) - done)

        pbar(done, len(loader),
             prefix=f"Epoch {epoch:02d}/{total_epochs} [train]",
             suffix=f"loss={total_loss / n:.4f} IoU={iou_sum / n:.3f} ETA {eta:.0f}s")

    return total_loss / max(n, 1), iou_sum / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, giou_crit, l1_crit, device, use_amp, tag="val"):
    model.eval()
    total_loss = 0
    iou_sum = 0
    n = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        images = batch["image"].to(device, non_blocking=True)
        tgt_norm = batch["bbox"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred_raw = model.forward_normalised(images)
            giou_loss = giou_crit(pred_raw, tgt_norm)
            l1_loss = l1_crit(pred_raw.clamp(0, 1), tgt_norm)
            loss = giou_loss + 0.5 * l1_loss

        bs = images.size(0)
        total_loss += loss.item() * bs
        iou_sum += mean_iou(pred_raw, tgt_norm) * bs
        n += bs

        done = step + 1
        eta = (time.time() - t0) / done * (len(loader) - done)

        pbar(done, len(loader), prefix=f"[{tag}]", suffix=f"ETA {eta:.0f}s")

    return total_loss / max(n, 1), iou_sum / max(n, 1)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        use_amp = True
        default_bs = 32
    else:
        device = torch.device("cpu")
        use_amp = False
        default_bs = 16
        print("No GPU found. Run check_gpu.py")

    batch_size = args.batch_size or default_bs
    sz = args.image_size

    print(f"Device={device} image_size={sz} batch={batch_size} AMP={use_amp}")

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name or f"loc_{args.freeze_blocks}", config=vars(args))

    full_ds = OxfordIIITPetDataset(root=args.data_root, split="trainval", transform=get_transform(sz, True), download=args.download)

    n_val = int(len(full_ds) * args.val_split)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))

    val_ds.dataset.transform = get_transform(sz, False)

    test_ds = OxfordIIITPetDataset(root=args.data_root, split="test", transform=get_transform(sz, False))

    kw = dict(collate_fn=collate_fn_bbox, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    print(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    model = VGG11Localizer(in_channels=3, dropout_p=args.dropout_p, image_size=sz, freeze_blocks=args.freeze_blocks, pretrained_clf=args.pretrained_clf).to(device)

    giou_crit = GIoULoss(reduction="mean")
    l1_crit = nn.SmoothL1Loss()

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.reg_head.parameters())

    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": args.lr / 10},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou = train_one_epoch(model, train_loader, giou_crit, l1_crit, optimizer, scaler, device, epoch, args.epochs, use_amp)
        vl_loss, vl_iou = evaluate(model, val_loader, giou_crit, l1_crit, device, use_amp, "val")

        scheduler.step()

        print(f"loss={tr_loss:.4f} IoU={tr_iou:.3f} | val_loss={vl_loss:.4f} IoU={vl_iou:.3f} | lr={get_lr(optimizer):.2e}")

        if vl_iou > best_iou:
            best_iou = vl_iou
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_iou": vl_iou, "args": vars(args)},
                       Path(args.checkpoint_dir) / "best_localizer.pth")

    ckpt = torch.load(Path(args.checkpoint_dir) / "best_localizer.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    ts_loss, ts_iou = evaluate(model, test_loader, giou_crit, l1_crit, device, use_amp, "test")

    print(f"FINAL TEST loss={ts_loss:.4f} mean_IoU={ts_iou:.3f}")


if __name__ == "__main__":
    main()