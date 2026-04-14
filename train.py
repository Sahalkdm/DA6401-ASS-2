

import argparse
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import GIoULoss
from losses.dice_loss import DiceLoss

import torch.nn.functional as F

# @torch.no_grad()
# def compute_mean_iou(pred_norm, tgt_norm, eps=1e-6):
#     p = pred_norm.clamp(0, 1)
#     px1 = p[:,0]-p[:,2]/2; px2 = p[:,0]+p[:,2]/2
#     py1 = p[:,1]-p[:,3]/2; py2 = p[:,1]+p[:,3]/2
#     tx1 = tgt_norm[:,0]-tgt_norm[:,2]/2; tx2 = tgt_norm[:,0]+tgt_norm[:,2]/2
#     ty1 = tgt_norm[:,1]-tgt_norm[:,3]/2; ty2 = tgt_norm[:,1]+tgt_norm[:,3]/2
#     iw  = (torch.min(px2,tx2)-torch.max(px1,tx1)).clamp(0)
#     ih  = (torch.min(py2,ty2)-torch.max(py1,ty1)).clamp(0)
#     inter = iw*ih
#     union = p[:,2].clamp(0)*p[:,3].clamp(0) + \
#             tgt_norm[:,2].clamp(0)*tgt_norm[:,3].clamp(0) - inter
#     return (inter/(union+eps)).mean().item()

@torch.no_grad()
def compute_mean_iou(pred_box, tgt_box, eps=1e-6):
    # No clamp(0, 1) here anymore because our coordinates are in actual pixels (0 to 224)!
    
    # Extract coordinates directly (Format: xmin, ymin, xmax, ymax)
    px1, py1, px2, py2 = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2], pred_box[:, 3]
    tx1, ty1, tx2, ty2 = tgt_box[:, 0], tgt_box[:, 1], tgt_box[:, 2], tgt_box[:, 3]
    
    # 1. Calculate Intersection
    # Find the coordinates of the intersection rectangle
    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)
    
    # Calculate intersection width and height (clamp to 0 to avoid negative areas)
    iw = (inter_x2 - inter_x1).clamp(min=0)
    ih = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = iw * ih
    
    # 2. Calculate Union
    # Area of predicted and target boxes
    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    tgt_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    
    union_area = pred_area + tgt_area - inter_area
    
    # 3. Calculate IoU
    iou = inter_area / (union_area + eps)
    
    return iou.mean().item()

@torch.no_grad()
def compute_dice(logits, targets, num_classes=3, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    scores = []
    for c in range(num_classes):
        p = (preds==c).float(); t = (targets==c).float()
        scores.append(((2*(p*t).sum()+smooth)/((p+t).sum()+smooth)).item())
    return float(np.mean(scores))

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from models.multitask import MultiTaskPerceptionModel
# Ensure this matches your actual import path for the dataset
from data.pets_dataset import OxfordIIITPetDataset 

# Dataset Wrapper
from torch.utils.data import Dataset

torch.backends.cudnn.benchmark = True

class MultitaskWrapper(Dataset):
    def __init__(self, base_subset, transform):
        self.base = base_subset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # 1. Get raw data from the base dataset
        data = self.base[idx]
        image_pil = data["image"]
        mask_pil = data["mask"]
        bbox = data["bbox"] # [cx, cy, w, h] normalized
        label = data["label"]

        # Convert PIL to numpy for Albumentations
        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil) if mask_pil is not None else None

        # 2. Build kwargs for Albumentations
        kwargs = {"image": image_np}
        if mask_np is not None:
            kwargs["mask"] = mask_np
            
        # CRITICAL FIX: Albumentations requires these keys to always exist!
        if bbox is not None:
            kwargs["bboxes"] = [bbox]
            kwargs["class_labels"] = [label]
        else:
            kwargs["bboxes"] = []
            kwargs["class_labels"] = []

        # 3. Apply the transform (This flips/resizes image, mask, AND bbox)
        out = self.transform(**kwargs)
        
        img_tensor = out["image"]
        mask_tensor = out.get("mask", None)
        
        # 4. Convert augmented YOLO bbox to absolute Pascal VOC [xmin, ymin, xmax, ymax]
        out_bbox = None
        if bbox is not None and len(out["bboxes"]) > 0:
            cx, cy, w, h = out["bboxes"][0]
            _, H, W = img_tensor.shape # Get final image dimensions (e.g., 224x224)
            
            xmin = (cx - w / 2) * W
            ymin = (cy - h / 2) * H
            xmax = (cx + w / 2) * W
            ymax = (cy + h / 2) * H
            out_bbox = [xmin, ymin, xmax, ymax]

        return {
            "image": img_tensor,
            "label": label,
            "bbox": out_bbox,
            "mask": mask_tensor
        }

# ─── Transforms ──────────────────────────────────────────────────────────────
def get_train_transform(sz):
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])) # Tell Albumentations it's YOLO!

def get_val_transform(sz):
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ─── Collate — handles all three tasks, skips missing annotations ─────────────
def collate_fn_multitask(batch):
    """Stack images always; bbox/mask are optional (None if not annotated)."""
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    # Bounding boxes — only include samples that have them
    bbox_indices = [i for i, b in enumerate(batch) if b.get("bbox") is not None]
    bboxes       = torch.tensor([batch[i]["bbox"] for i in bbox_indices],
                                dtype=torch.float32) if bbox_indices else None

    # Masks — only include samples that have them
    mask_indices = [i for i, b in enumerate(batch) if b.get("mask") is not None]
    masks = []
    for i in mask_indices:
        m = batch[i]["mask"]
        if isinstance(m, torch.Tensor):
            m = m.long()
        else:
            m = torch.from_numpy(np.array(m)).long()
        m = (m - 1).clamp(0, 2)   # remap trimap 1,2,3 → 0,1,2
        masks.append(m)
    masks = torch.stack(masks) if masks else None

    return {
        "image":        images,
        "label":        labels,
        "bbox":         bboxes,
        "bbox_indices": bbox_indices,
        "mask":         masks,
        "mask_indices": mask_indices,
    }

# ─── Evaluation Helper ────────────────────────────────────────────────────────
def calculate_iou(pred_box, true_box):
    """Simple IoU for bounding boxes [x_min, y_min, x_max, y_max]"""
    x1 = torch.max(pred_box[:, 0], true_box[:, 0])
    y1 = torch.max(pred_box[:, 1], true_box[:, 1])
    x2 = torch.min(pred_box[:, 2], true_box[:, 2])
    y2 = torch.min(pred_box[:, 3], true_box[:, 3])
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1])
    box2_area = (true_box[:, 2] - true_box[:, 0]) * (true_box[:, 3] - true_box[:, 1])
    union = box1_area + box2_area - intersection
    return (intersection / (union + 1e-6)).mean().item()

def calculate_dice(pred_mask, true_mask, num_classes=3):
    """Simple multiclass Dice score"""
    pred_labels = torch.argmax(pred_mask, dim=1)
    dice_scores = []
    for cls in range(num_classes):
        pred_c = (pred_labels == cls)
        true_c = (true_mask == cls)
        intersection = (pred_c & true_c).sum((1, 2)).float()
        union = pred_c.sum((1, 2)).float() + true_c.sum((1, 2)).float()
        dice = 2.0 * intersection / (union + 1e-6)
        dice_scores.append(dice.mean().item())
    return sum(dice_scores) / num_classes


@torch.no_grad()
def evaluate(model, loader, losses, device, use_amp, lambdas, split_name="val"):
    model.eval()
    all_preds_cls, all_labels_cls = [], []
    total_iou, total_dice = 0.0, 0.0
    iou_count, dice_count = 0, 0
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Evaluating [{split_name}]"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            
            # Classification
            all_preds_cls.extend(outputs['classification'].argmax(1).cpu().numpy())
            all_labels_cls.extend(labels.cpu().numpy())
            loss_cls = losses['cls'](outputs['classification'], labels)
            
            # Localization
            loss_loc = torch.tensor(0.0, device=device)
            if batch["bbox_indices"]:
                bboxes = batch["bbox"].to(device)
                pred_bboxes = outputs['localization'][batch["bbox_indices"]]
                loss_loc = losses['loc'](pred_bboxes, bboxes)
                total_iou += calculate_iou(pred_bboxes, bboxes)
                iou_count += 1
                
            # Segmentation
            loss_seg = torch.tensor(0.0, device=device)
            if batch["mask_indices"]:
                masks = batch["mask"].to(device)
                pred_masks = outputs['segmentation'][batch["mask_indices"]]
                loss_seg = losses['seg'](pred_masks, masks)
                total_dice += calculate_dice(pred_masks, masks)
                dice_count += 1
                
            batch_loss = (lambdas['cls'] * loss_cls) + (lambdas['loc'] * loss_loc) + (lambdas['seg'] * loss_seg)
            total_loss += batch_loss.item()

    acc = sum(p == l for p, l in zip(all_preds_cls, all_labels_cls)) / len(all_labels_cls)
    f1 = f1_score(all_labels_cls, all_preds_cls, average="macro")
    iou = total_iou / max(1, iou_count)
    dice = total_dice / max(1, dice_count)

    return {"loss": total_loss / len(loader), "acc": acc, "f1": f1, "iou": iou, "dice": dice}


# ─── Main Training Loop ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./pets_data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sz", type=int, default=112)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    # W&B
    wandb.init(project="da6401_assignment2", name="multitask-finetuning", config=vars(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────
    # full_ds = OxfordIIITPetDataset(
    #     root=args.data_root, split="trainval",
    #     transform=get_train_transform(args.sz), download=args.download,
    # )
    # n_val   = int(len(full_ds) * args.val_split)
    # n_train = len(full_ds) - n_val
    # train_ds, val_ds = random_split(
    #     full_ds, [n_train, n_val],
    #     generator=torch.Generator().manual_seed(args.seed),
    # )
    # val_ds.dataset.transform = get_val_transform(args.sz)

    # test_ds = OxfordIIITPetDataset(
    #     root=args.data_root, split="test",
    #     transform=get_val_transform(args.sz),
    # )

    # kw = dict(collate_fn=collate_fn_multitask, num_workers=args.num_workers,
    #           pin_memory=(device.type=="cuda"),
    #           persistent_workers=(args.num_workers>0))
              
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **kw)
    # val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw)
    # test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, **kw)

    # ── Model ────────────────────────────────────────────────────────────
    # ── Dataset ──────────────────────────────────────────────────────────
    # Load the base dataset with NO transforms (so we get raw PIL images and raw bboxes)
    full_ds_raw = OxfordIIITPetDataset(
        root=args.data_root, split="trainval",
        transform=None, download=args.download,
    )
    
    # Split the raw data
    n_val   = int(len(full_ds_raw) * args.val_split)
    n_train = len(full_ds_raw) - n_val
    train_raw, val_raw = random_split(
        full_ds_raw, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Load test set raw
    test_raw = OxfordIIITPetDataset(
        root=args.data_root, split="test",
        transform=None,
    )

    # Wrap them in our custom wrapper to apply specific transforms and fix bbox coordinates!
    train_ds = MultitaskWrapper(train_raw, get_train_transform(args.sz))
    val_ds   = MultitaskWrapper(val_raw, get_val_transform(args.sz))
    test_ds  = MultitaskWrapper(test_raw, get_val_transform(args.sz))

    kw = dict(collate_fn=collate_fn_multitask, num_workers=args.num_workers,
              pin_memory=(device.type=="cuda"),
              persistent_workers=(args.num_workers>0))
              
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, **kw)
    
    model = MultiTaskPerceptionModel(image_size=args.sz, device=device).to(device)

    # ── Losses & Optimizer ───────────────────────────────────────────────
    losses = {
        'cls': nn.CrossEntropyLoss(),
        'loc': nn.MSELoss(),
        'seg': nn.CrossEntropyLoss()
    }
    lambdas = {'cls': 1.0, 'loc': 0.01, 'seg': 1.0} # Loss weighting factors
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_score = 0.0

    # ── Training Loop ────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                
                loss_cls = losses['cls'](outputs['classification'], labels)
                
                # Check for bboxes
                if batch["bbox_indices"]:
                    bboxes = batch["bbox"].to(device)
                    pred_bboxes = outputs['localization'][batch["bbox_indices"]]
                    loss_loc = losses['loc'](pred_bboxes, bboxes)
                else:
                    loss_loc = torch.tensor(0.0, device=device)
                    
                # Check for masks
                if batch["mask_indices"]:
                    masks = batch["mask"].to(device)
                    pred_masks = outputs['segmentation'][batch["mask_indices"]]
                    loss_seg = losses['seg'](pred_masks, masks)
                else:
                    loss_seg = torch.tensor(0.0, device=device)

                total_loss = (lambdas['cls'] * loss_cls) + (lambdas['loc'] * loss_loc) + (lambdas['seg'] * loss_seg)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()
            pbar.set_postfix({"Loss": total_loss.item()})

        # ── Validation ───────────────────────────────────────────────────────
        vl = evaluate(model, val_loader, losses, device, use_amp, lambdas, "val")
        
        combined = vl["f1"] + vl["iou"] + vl["dice"]
        
        wandb.log({
            "epoch": epoch, "train_loss": running_loss/len(train_loader), 
            "val_loss": vl["loss"], "val_f1": vl["f1"], "val_iou": vl["iou"], 
            "val_dice": vl["dice"], "combined_score": combined
        })
        
        if combined > best_score:
            best_score = combined
            ckpt_path  = Path(args.checkpoint_dir) / "best_multitask.pth"
            torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                            val_f1=vl["f1"], val_iou=vl["iou"], val_dice=vl["dice"],
                            combined=combined, args=vars(args)), ckpt_path)
            print(f"  ✓ Checkpoint  f1={vl['f1']:.3f} iou={vl['iou']:.3f} "
                  f"dice={vl['dice']:.3f}  (combined={combined:.3f})\n")
        else:
            print(f"  - Val Info    f1={vl['f1']:.3f} iou={vl['iou']:.3f} dice={vl['dice']:.3f}\n")

    # ── Final Test Evaluation ─────────────────────────────────────────────
    print("=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    ckpt = torch.load(Path(args.checkpoint_dir)/"best_multitask.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    ts = evaluate(model, test_loader, losses, device, use_amp, lambdas, "test")
    print(f"\nTest  F1   (classification) : {ts['f1']:.4f}")
    print(f"Test  IoU  (localisation)   : {ts['iou']:.4f}")
    print(f"Test  Dice (segmentation)   : {ts['dice']:.4f}")
    print(f"Test  Acc  (classification) : {ts['acc']:.4f}")
    
    wandb.log({"test_f1": ts["f1"], "test_iou": ts["iou"], "test_dice": ts["dice"]})
    wandb.finish()

if __name__ == "__main__":
    main()