import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary (multi-label) focal loss with logits.
    - alpha: float or Tensor (positives weight). e.g., 0.25 or shape [num_tasks] for per-task.
             If None, no alpha weighting.
    - gamma: focusing parameter (>=0).
    - sample_weight: optional per-sample/per-element weight (same shape as inputs/targets or broadcastable).
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'): # 0.35 -> 0.25
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, sample_weight=None):
        # ensure float
        inputs = inputs.float()
        targets = targets.float()

        # 1) elementwise BCE with logits (stable)
        ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # = -log(p_t)

        # 2) p_t from CE (stable): p_t = exp(-CE)
        pt = torch.exp(-ce)

        # 3) focal modulation
        focal_factor = (1.0 - pt) ** self.gamma

        # 4) alpha weighting (positives only). allow scalar or tensor (broadcast)
        if self.alpha is not None:
            if not torch.is_tensor(self.alpha):
                alpha_pos = torch.tensor(self.alpha, dtype=targets.dtype, device=targets.device)
                alpha_neg = 1.0 - alpha_pos
            else:
                # assume alpha is "positives weight"; negatives weight = 1 - alpha
                alpha_pos = self.alpha.to(device=targets.device, dtype=targets.dtype)
                alpha_neg = 1.0 - alpha_pos
            alpha_t = alpha_pos * targets + alpha_neg * (1.0 - targets)
        else:
            alpha_t = 1.0

        loss = alpha_t * focal_factor * ce

        # 5) optional sample-wise weights (e.g., class-balancing, survey weights)
        if sample_weight is not None:
            loss = loss * sample_weight

        # 6) reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
