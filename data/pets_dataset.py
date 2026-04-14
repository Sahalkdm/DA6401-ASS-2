"""Dataset skeleton for Oxford-IIIT Pet.
"""

from torch.utils.data import Dataset

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
 
import numpy as np
from PIL import Image

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    _SPLIT_FILES = {
        "trainval": "trainval.txt",
        "test": "test.txt",
    }
 
    def __init__(
        self, root: str, split: str = "trainval", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
 
        if download:
            self._download()
 
        self.images_dir = self.root / "images"
        self.annots_dir = self.root / "annotations"
        self.xmls_dir = self.annots_dir / "xmls"
        self.masks_dir = self.annots_dir / "trimaps"
 
        if not self.images_dir.exists():
            raise RuntimeError(
                f"Dataset not found at '{root}'. Set download=True."
            )
 
        split_path = self.annots_dir / self._SPLIT_FILES[split]
        self.samples: List[Dict] = []
        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                self.samples.append({
                    "image_name": parts[0],
                    "label": int(parts[1]) - 1,  # 0-indexed
                })
 
        self._build_class_names()
 
    @property
    def num_classes(self):
        return 37
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        s = self.samples[idx]
        name = s["image_name"]
        label = s["label"]
 
        img_path = self.images_dir / f"{name}.jpg"
        image = Image.open(img_path).convert("RGB")
        w_img, h_img = image.size
 
        bbox = None
        xml_path = self.xmls_dir / f"{name}.xml"
        if xml_path.exists():
            bbox = self._parse_bbox_xml(xml_path, w_img, h_img)
 
        mask = None
        mask_path = self.masks_dir / f"{name}.png"
        if mask_path.exists():
            mask = Image.open(mask_path)
 
        if self.transform is not None:
            image_np = np.array(image)
            if mask is not None:
                mask_np = np.array(mask)
                out = self.transform(image=image_np, mask=mask_np)
                image = out["image"]
                mask = out["mask"]
            else:
                image = self.transform(image=image_np)["image"]
 
        if self.target_transform is not None:
            label = self.target_transform(label)
 
        return {"image": image, "label": label, "bbox": bbox, "mask": mask}
 
    def _parse_bbox_xml(self, xml_path, img_w, img_h):
        try:
            root = ET.parse(xml_path).getroot()
            obj  = root.find("object")
            if obj is None:
                return None
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text)
            ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text)
            ymax = float(bb.find("ymax").text)
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            return [cx, cy, bw, bh]
        except Exception:
            return None
 
    def _build_class_names(self):
        name_map: Dict[int, str] = {}
        for s in self.samples:
            lbl = s["label"]
            if lbl not in name_map:
                name_map[lbl] = "_".join(s["image_name"].split("_")[:-1])
        self.class_names = [name_map.get(i, str(i)) for i in range(37)]
 
    def _download(self):
        try:
            import torchvision.datasets as tvd
            tvd.OxfordIIITPets(root=str(self.root), download=True)
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")


