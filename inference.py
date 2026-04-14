"""Inference and evaluation
"""

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

from models.multitask import MultiTaskPerceptionModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEG_LABELS = {0: "Pet (foreground)", 1: "Background", 2: "Boundary"}


def get_transform(sz=224):
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_model(ckpt_path, image_size=224, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    args = ckpt.get("args", {})

    model = MultiTaskPerceptionModel(
        num_breeds=args.get("num_breeds", 37),
        seg_classes=args.get("seg_classes", 3),
        image_size=image_size,
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_single(model, image_path, image_size=224, device="cpu"):
    transform = get_transform(image_size)

    img = np.array(Image.open(image_path).convert("RGB"))
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    out = model(tensor)

    cls_probs = F.softmax(out["classification"], dim=1)[0]
    cls_label = cls_probs.argmax().item()
    cls_conf = cls_probs.max().item()

    bbox_px = out["localization"][0].cpu().tolist()

    seg_logits = out["segmentation"][0]
    seg_mask = seg_logits.argmax(dim=0).cpu().numpy()

    return {
        "class_id": cls_label,
        "confidence": cls_conf,
        "bbox_pixels": bbox_px,
        "seg_mask": seg_mask,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--data_root", type=str, default="./pets_data")
    p.add_argument("--eval_split", type=str, default=None, choices=["trainval", "test"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}")

    model = load_model(args.checkpoint, args.image_size, device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {total} parameters")

    if args.image:
        result = predict_single(model, args.image, args.image_size, device)

        print(f"Classification: class_id={result['class_id']} confidence={result['confidence']:.3f}")

        cx, cy, w, h = result["bbox_pixels"]
        print(f"Bounding box: cx={cx:.1f} cy={cy:.1f} w={w:.1f} h={h:.1f}")

        mask = result["seg_mask"]
        unique, counts = np.unique(mask, return_counts=True)

        print("Segmentation:", end=" ")
        for u, c in zip(unique, counts):
            print(f"{SEG_LABELS.get(u, u)}={c / mask.size:.1%}", end=" ")
        print()

    if args.eval_split:
        from torch.utils.data import DataLoader
        from sklearn.metrics import f1_score
        from data.pets_dataset import OxfordIIITPetDataset
        from train import collate_fn_multitask, compute_mean_iou, compute_dice

        transform = get_transform(args.image_size)

        ds = OxfordIIITPetDataset(
            root=args.data_root,
            split=args.eval_split,
            transform=transform,
        )

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            collate_fn=collate_fn_multitask,
            num_workers=args.num_workers,
        )

        all_preds = []
        all_labels = []

        iou_sum = 0
        iou_n = 0
        dice_sum = 0
        dice_n = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                out = model(images)

                all_preds.extend(out["classification"].argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if batch["bbox"] is not None:
                    idx = batch["bbox_indices"]
                    pred = out["localization"][idx] # / args.image_size
                    tgt = batch["bbox"].to(device)

                    iou_sum += compute_mean_iou(pred, tgt)
                    iou_n += 1

                if batch["mask"] is not None:
                    idx = batch["mask_indices"]
                    pred = out["segmentation"][idx]
                    tgt = batch["mask"].to(device)

                    dice_sum += compute_dice(pred, tgt)
                    dice_n += 1

        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        iou = iou_sum / max(iou_n, 1)
        dice = dice_sum / max(dice_n, 1)

        print("\n" + "=" * 50)
        print(f"Evaluation on {args.eval_split} split")
        print(f"Classification: Acc={acc:.4f} Macro-F1={f1:.4f}")
        print(f"Localization: Mean IoU={iou:.4f}")
        print(f"Segmentation: Dice={dice:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
