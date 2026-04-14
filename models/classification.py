"""Classification components
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes= 37, in_channels= 3, dropout_p=0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
 
        # Encoder (shared convolutional backbone) 
        self.encoder = VGG11Encoder(in_channels=in_channels)
 
        # AdaptiveAvgPool makes the head input-size agnostic
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
 
        self.classifier = nn.Sequential(
            # FC-1
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
 
            # FC-2
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
 
            # FC-3 (output layer — no dropout, no activation)
            nn.Linear(4096, num_classes),
        )
 
        # Initialise FC layer weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Normal init for Linear layers. """
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        # TODO: Implement forward pass.

        # Extract features via the VGG11
        features = self.encoder(x, return_features=False) 
 
        # Adaptive average pool (ensures 7×7 regardless of input size)
        pooled = self.avgpool(features)
 
        # Flatten to 1-D vector per sample
        flat = torch.flatten(pooled, start_dim=1)
 
        # FC head with Dropout regularisation
        logits = self.classifier(flat)
 
        return logits
    
        raise NotImplementedError("Implement VGG11Classifier.forward")