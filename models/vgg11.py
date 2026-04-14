"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout

class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()
 
        # Helper to build Conv -> BN -> ReLU triple
        def conv_bn_relu(in_ch, out_ch, kernel=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
 
        #  Convolutional Blocks 
 
        self.block1 = nn.Sequential(
            conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
 
        self.block2 = nn.Sequential(
            conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
 
        self.block3 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
 
        self.block4 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
 
        self.block5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
 
        # Weight initialisation 
        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) init for Conv layers; constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight) 
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: Implement forward pass.
        f1 = self.block1(x)   # [B,  64, 112, 112]
        f2 = self.block2(f1)  # [B, 128,  56,  56]
        f3 = self.block3(f2)  # [B, 256,  28,  28]
        f4 = self.block4(f3)  # [B, 512,  14,  14]
        f5 = self.block5(f4)  # [B, 512,   7,   7]
 
        if return_features:
            features = {
                "block1": f1,
                "block2": f2,
                "block3": f3,
                "block4": f4,
                "block5": f5,
            }
            return f5, features
 
        return f5
    
        raise NotImplementedError("Implement VGG11Encoder.forward")