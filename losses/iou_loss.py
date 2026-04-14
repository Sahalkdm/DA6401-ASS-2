import torch
import torch.nn as nn
 
 
class IoULoss(nn.Module):
    """Standard IoU loss — (cx,cy,w,h) format.  Required by autograder."""
 
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be none/mean/sum, got '{reduction}'")
        self.eps = eps
        self.reduction = reduction
 
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes:   [B,4] (cx,cy,w,h)
            target_boxes: [B,4] (cx,cy,w,h)
        Returns:
            Scalar loss (mean/sum) or [B] (none).
        """
        # Convert to corners
        p_x1 = pred_boxes[:,0] - pred_boxes[:,2]/2
        p_y1 = pred_boxes[:,1] - pred_boxes[:,3]/2
        p_x2 = pred_boxes[:,0] + pred_boxes[:,2]/2
        p_y2 = pred_boxes[:,1] + pred_boxes[:,3]/2
 
        t_x1 = target_boxes[:,0] - target_boxes[:,2]/2
        t_y1 = target_boxes[:,1] - target_boxes[:,3]/2
        t_x2 = target_boxes[:,0] + target_boxes[:,2]/2
        t_y2 = target_boxes[:,1] + target_boxes[:,3]/2
 
        # Intersection
        inter_w = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0)
        inter_h = (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
        inter = inter_w * inter_h
 
        pred_area = pred_boxes[:,2].clamp(0) * pred_boxes[:,3].clamp(0)
        target_area = target_boxes[:,2].clamp(0) * target_boxes[:,3].clamp(0)
        union = pred_area + target_area - inter
 
        iou = inter / (union + self.eps)
        loss = 1.0 - iou
 
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss
 
    def extra_repr(self): return f"eps={self.eps}, reduction='{self.reduction}'"
 
 
class GIoULoss(nn.Module):
 
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be none/mean/sum, got '{reduction}'")
        self.eps = eps
        self.reduction = reduction
 
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        p_x1 = pred_boxes[:,0] - pred_boxes[:,2]/2
        p_y1 = pred_boxes[:,1] - pred_boxes[:,3]/2
        p_x2 = pred_boxes[:,0] + pred_boxes[:,2]/2
        p_y2 = pred_boxes[:,1] + pred_boxes[:,3]/2
 
        t_x1 = target_boxes[:,0] - target_boxes[:,2]/2
        t_y1 = target_boxes[:,1] - target_boxes[:,3]/2
        t_x2 = target_boxes[:,0] + target_boxes[:,2]/2
        t_y2 = target_boxes[:,1] + target_boxes[:,3]/2
 
        #  Intersection 
        inter_w = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0)
        inter_h = (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
        inter   = inter_w * inter_h
 
        pred_area   = pred_boxes[:,2].clamp(0) * pred_boxes[:,3].clamp(0)
        target_area = target_boxes[:,2].clamp(0) * target_boxes[:,3].clamp(0)
        union = pred_area + target_area - inter
 
        iou = inter / (union + self.eps)                       # [B]
 
        #  Enclosing box 
        enc_x1 = torch.min(p_x1, t_x1)
        enc_y1 = torch.min(p_y1, t_y1)
        enc_x2 = torch.max(p_x2, t_x2)
        enc_y2 = torch.max(p_y2, t_y2)
 
        enc_area = ((enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0))   
 
        #  GIoU 
        giou = iou - (enc_area - union) / (enc_area + self.eps)
        loss = 1.0 - giou                                      
 
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss
 
    def extra_repr(self): return f"eps={self.eps}, reduction='{self.reduction}'"
 