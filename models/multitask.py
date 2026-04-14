"""Unified multi-task model
"""
# https://drive.google.com/file/d/1iq30Vc-d2e3wBYx9ydqL_bdqCRlsz7ys/view?usp=sharing
# https://drive.google.com/file/d/1CUz43aWejS52xCrgPqwt3YbXbnesCfmu/view?usp=sharing
# https://drive.google.com/file/d/1qHV8TNQsVgxsiUjxFmm4Vg9Cnu0vMxm1/view?usp=sharing

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.classification import VGG11Classifier 
from models.localization import VGG11Localizer 
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", 
                 unet_path: str = "unet.pth", device='cpu', image_size: int = 224):
        
        super().__init__() 
        self.image_size = image_size

        # Download weights
        import gdown
        import os
        if not os.path.exists(classifier_path): gdown.download(id="1iq30Vc-d2e3wBYx9ydqL_bdqCRlsz7ys", output=classifier_path, quiet=False)
        if not os.path.exists(localizer_path): gdown.download(id="1CUz43aWejS52xCrgPqwt3YbXbnesCfmu", output=localizer_path, quiet=False)
        if not os.path.exists(unet_path): gdown.download(id="1qHV8TNQsVgxsiUjxFmm4Vg9Cnu0vMxm1", output=unet_path, quiet=False)

        # SHARED ENCODER & POOLER
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # EXTRACT HEADS FROM YOUR MODELS
        # Classifier Head
        self.classifier = VGG11Classifier(num_classes=num_breeds).classifier
        
        # Localizer Head
        self.reg_head = VGG11Localizer().reg_head

        # Segmentation Decoder Components
        # We instantiate a temporary UNet just to extract its decoder layers
        tmp_unet = VGG11UNet(num_classes=seg_classes)
        self.bottleneck_drop = tmp_unet.bottleneck_drop
        self.d5 = tmp_unet.d5
        self.d4 = tmp_unet.d4
        self.d3 = tmp_unet.d3
        self.d2 = tmp_unet.d2
        self.d1 = tmp_unet.d1
        self.output_conv = tmp_unet.output_conv
        del tmp_unet # Free memory

        # 4. LOAD & INJECT WEIGHTS
        print("Loading weights into PyTorch...")
        cls_ckpt = torch.load(classifier_path, map_location=torch.device('cpu'))
        loc_ckpt = torch.load(localizer_path, map_location=torch.device('cpu'))
        seg_ckpt = torch.load(unet_path, map_location=torch.device('cpu'))

        cls_state = cls_ckpt.get('model_state_dict', cls_ckpt)
        loc_state = loc_ckpt.get('model_state_dict', loc_ckpt)
        seg_state = seg_ckpt.get('model_state_dict', seg_ckpt)

        print("Injecting weights into the Unified Model...")
        # Load the shared backbone
        self.encoder.load_state_dict(cls_state, strict=False)
        
        # Load the individual heads (strict=False ensures they only pick up their specific keys)
        self.classifier.load_state_dict(cls_state, strict=False)
        self.reg_head.load_state_dict(loc_state, strict=False)
        
        # Load the UNet components
        self.load_state_dict(seg_state, strict=False) 
        
        print(" -> Multi-Task Model successfully assembled and loaded!")


    def forward(self, x: torch.Tensor):
        # SHARED ENCODER
        bottleneck, skips = self.encoder(x, return_features=True)

        # PREPARE POOLED FEATURES
        pooled = self.avgpool(bottleneck)
        flat = torch.flatten(pooled, start_dim=1)

        # CLASSIFICATION BRANCH
        cls_logits = self.classifier(flat)

        # LOCALIZATION BRANCH
        # Match your VGG11Localizer forward logic (Clamp & Scale)
        raw_coords = self.reg_head(flat)
        loc_coords = raw_coords.clamp(0, 1)

        # SEGMENTATION BRANCH
        # Match your VGG11UNet forward logic exactly
        b = self.bottleneck_drop(bottleneck)
        x5 = self.d5(b,  skips["block4"])
        x4 = self.d4(x5, skips["block3"])
        x3 = self.d3(x4, skips["block2"])
        x2 = self.d2(x3, skips["block1"])
        x1 = self.d1(x2)
        seg_mask = self.output_conv(x1)

        # UNIFIED OUTPUT 
        return {
            'classification': cls_logits,
            'localization': loc_coords,
            'segmentation': seg_mask
        }