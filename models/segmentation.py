"""Segmentation model
"""


import torch
import torch.nn as nn
from typing import Optional
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

 
#  Decoder building blocks
 
def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv 3x3 → BatchNorm → ReLU (padding=1 keeps spatial size)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
 
 
class DecoderBlock(nn.Module):
    
    def __init__(self, in_ch, skip_ch, out_ch):
        
        super().__init__()
 
        # kernel=2, stride=2 → exact ×2 upsample, no output_padding needed
        self.up = nn.ConvTranspose2d(
            in_ch, in_ch // 2,
            kernel_size=2, stride=2, bias=False
        )
 
        fused_ch = in_ch // 2 + skip_ch
        self.conv = nn.Sequential(
            conv_bn_relu(fused_ch, out_ch),
            conv_bn_relu(out_ch,   out_ch),
        )
 
    def forward(self, x, skip):
        
        x = self.up(x)
 
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
 
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
 
 
class FinalDecoderBlock(nn.Module):
   
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.conv = nn.Sequential(
            conv_bn_relu(out_ch, out_ch),
            conv_bn_relu(out_ch, out_ch),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(x)

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, pretrained_clf=None, freeze_encoder=False):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
 
        # Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)
 
        if pretrained_clf is not None:
            self._load_encoder(pretrained_clf)
 
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("[UNet] Encoder frozen (strict feature extractor)")
 
        # Bottleneck dropout
        self.bottleneck_drop = CustomDropout(p=dropout_p)
 
        # Decoder (symmetric expansive path)
        # Each DecoderBlock: TransposedConv(×2) -> concat(skip) -> 2×Conv-BN-ReLU
        #
        # Channel accounting:
        #   d5: in=512,  skip=512(block4), out=512
        #   d4: in=512,  skip=256(block3), out=256
        #   d3: in=256,  skip=128(block2), out=128
        #   d2: in=128,  skip= 64(block1), out= 64
        #   d1: in= 64,  no skip,          out= 32
        self.d5 = DecoderBlock(in_ch=512, skip_ch=512, out_ch=512)
        self.d4 = DecoderBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.d3 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.d2 = DecoderBlock(in_ch=128, skip_ch=64,  out_ch=64)
        self.d1 = FinalDecoderBlock(in_ch=64, out_ch=32)
 
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)
 
        # Weight init for decoder
        self._init_decoder_weights()
        pass

    # Weight and Bias init
    def _load_encoder(self, ckpt_path: str):
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        enc_state = {
            k.replace("encoder.", "", 1): v
            for k, v in state.items() if k.startswith("encoder.")
        }
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
        print(f"[UNet] Loaded encoder weights from '{ckpt_path}'")
        if missing: print(f"Missing : {missing}")
        if unexpected: print(f"Unexpected: {unexpected}")
 
    def _init_decoder_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.encoder:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # TODO: Implement forward pass.
        # Encoder
        bottleneck, skips = self.encoder(x, return_features=True)
 
        #  Bottleneck dropout 
        b = self.bottleneck_drop(bottleneck)
 
        #  Decoder- upsampling with skip connections 
        x5 = self.d5(b,  skips["block4"])      # [B, 512, 14, 14]
        x4 = self.d4(x5, skips["block3"])      # [B, 256, 28, 28]
        x3 = self.d3(x4, skips["block2"])      # [B, 128, 56, 56]
        x2 = self.d2(x3, skips["block1"])      # [B,  64, 112, 112]
        x1 = self.d1(x2)                       # [B,  32, 224, 224]
 
        #  Output head 
        logits = self.output_conv(x1)
        return logits
        raise NotImplementedError("Implement VGG11UNet.forward")
