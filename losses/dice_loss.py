import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class DiceLoss(nn.Module):
    """Multi-class soft Dice loss.
 
    Computes Dice loss per class then averages (macro Dice).
    Uses soft predictions (probabilities) for differentiability.
    """
 
    def __init__(self, num_classes: int = 3, smooth: float = 1.0, ignore_index: int = -1):
        
        super().__init__()
        self.num_classes  = num_classes
        self.smooth       = smooth
        self.ignore_index = ignore_index
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W] raw logits (before softmax).
            targets: [B, H, W]    integer class labels.
        Returns:
            Scalar Dice loss averaged over classes and batch.
        """
        B, C, H, W = logits.shape
 
        # Soft predictions via softmax
        probs = F.softmax(logits, dim=1)             # [B, C, H, W]
 
        t = targets.clone()
        if self.ignore_index >= 0:
            t[t == self.ignore_index] = 0
        one_hot = F.one_hot(t.clamp(0, C-1), num_classes=C)  # [B,H,W,C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()         # [B,C,H,W]
 
        # Flatten spatial dims for dot-product
        probs_flat   = probs.view(B, C, -1)      # [B, C, H*W]
        one_hot_flat = one_hot.view(B, C, -1)    # [B, C, H*W]
 
        # Dice per class per sample
        intersection = (probs_flat * one_hot_flat).sum(dim=2)   # [B, C]
        union  = probs_flat.sum(dim=2) + one_hot_flat.sum(dim=2)  # [B, C]
 
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
 
        # Average over classes and batch
        dice_loss = 1.0 - dice_per_class.mean()
        return dice_loss