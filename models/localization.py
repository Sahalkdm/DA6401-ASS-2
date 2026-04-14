"""Localization modules """

import torch
import torch.nn as nn
from typing import Optional

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class RegressionHead(nn.Module):
    
    def __init__(self, in_features: int = 25088, dropout_p: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            # FC-1
            nn.Linear(in_features, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            # FC-2
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            # Output — raw values, NO sigmoid
            nn.Linear(1024, 4),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        output_layer = self.head[-1]
        nn.init.zeros_(output_layer.weight)
        output_layer.bias.data.copy_(torch.tensor([0.5, 0.5, 0.5, 0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer."""

    def __init__(
        self,
        in_channels= 3,
        dropout_p= 0.5,
        image_size= 224,
        freeze_blocks= "1-3",
        pretrained_clf = None,
    ):
        super().__init__()
        self.image_size = image_size

        self.encoder = VGG11Encoder(in_channels=in_channels)
        if pretrained_clf is not None:
            self._load_encoder_from_classifier(pretrained_clf)
        self._apply_freeze(freeze_blocks)

        self.avgpool  = nn.AdaptiveAvgPool2d((7, 7))
        self.reg_head = RegressionHead(512 * 7 * 7, dropout_p)

    def _load_encoder_from_classifier(self, ckpt_path: str):
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        enc_state = {k.replace("encoder.", "", 1): v
                     for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
        print(f"[Localizer] Loaded encoder from '{ckpt_path}'")
        if missing: print(f"Missing: {missing}")
        if unexpected: print(f"Unexpected: {unexpected}")

    def _apply_freeze(self, strategy: str):
        
        if strategy == "none":
            for p in self.encoder.parameters(): p.requires_grad = True
            print("[Localizer] Freeze: none (full fine-tuning)")
        elif strategy == "1-3":
            for b in [self.encoder.block1, self.encoder.block2, self.encoder.block3]:
                for p in b.parameters(): p.requires_grad = False
            for b in [self.encoder.block4, self.encoder.block5]:
                for p in b.parameters(): p.requires_grad = True
            n = sum(p.numel() for bl in [self.encoder.block1,
                    self.encoder.block2, self.encoder.block3]
                    for p in bl.parameters())
            print(f"[Localizer] Freeze: 1-3  ({n:,} params frozen)")
        elif strategy == "all":
            for p in self.encoder.parameters(): p.requires_grad = False
            print("[Localizer] Freeze: all (strict feature extractor)")
        else:
            raise ValueError(f"freeze_blocks must be none/1-3/all, got '{strategy}'")

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.encoder(x, return_features=False)
        pooled = self.avgpool(feats)
        return torch.flatten(pooled, start_dim=1)

    def forward_normalised(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.reg_head(self._features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        norm = self.forward_normalised(x).clamp(0, 1)
        return norm * self.image_size
