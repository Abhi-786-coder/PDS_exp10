"""
src/focal_loss.py
Phase 3B: Per-Endpoint Focal Loss with NaN Masking.

Implements PerEndpointFocalLoss for multi-task toxicity prediction.
Each of the 12 Tox21 endpoints has its own alpha weight derived from
its positive class prevalence (toxic compound frequency).

Key design decisions (from roadmap):
  - Focal Loss (γ=2) is preferred over BCE + class_weight='balanced'
    for severely imbalanced endpoints (e.g. NR-AR-LBD: 2.8% toxic).
  - Per-endpoint alpha weights prevent the model from averaging out
    the hard negatives across endpoints of different difficulty.
  - NaN labels (unknown tests) are masked BEFORE loss calculation.
    The mask multiplies the per-sample loss by 0 for unknown labels,
    ensuring the model never learns from absent experimental data.

Reference: Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PerEndpointFocalLoss(nn.Module):
    """
    Multi-task Focal Loss with per-endpoint alpha weights and NaN masking.

    Args:
        pos_weights  : Dict mapping endpoint name → positive class count (or ratio).
                       Used to derive per-endpoint alpha (class-frequency weighting).
        target_names : Ordered list of endpoint names (must match model output order).
        gamma        : Focal loss exponent. Default 2.0 (Lin et al. 2017).

    Forward Args:
        logits  : (batch, 12) raw model outputs (before sigmoid).
        targets : (batch, 12) float labels. Use -1 for unknown/NaN labels.

    Returns:
        Scalar loss (mean over valid labels).
    """

    def __init__(self, pos_weights: dict, target_names: list, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.target_names = target_names

        # Per-endpoint alpha: derived from class imbalance ratio
        # alpha = n_pos / (n_pos + n_neg) — higher alpha → more weight on toxic class
        raw = torch.tensor([float(pos_weights[t]) for t in target_names])
        self.register_buffer('alpha', (raw / (raw + 1.0)).unsqueeze(0))  # shape (1, 12)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 12) — raw model outputs, NOT sigmoid-activated.
            targets : (B, 12) — ground truth. -1 = unknown (will be masked).

        Returns:
            Scalar focal loss averaged over all known (non-masked) labels.
        """
        # Build mask: 1 where label is known (0 or 1), 0 where unknown (-1)
        mask = (targets != -1).float()

        # Replace -1 with 0.0 so sigmoid/BCE doesn't produce NaN
        targets_clean = targets.clone().float()
        targets_clean[targets == -1] = 0.0

        # Expand alpha to batch size
        alpha = self.alpha.to(logits.device).expand_as(logits)  # (B, 12)

        # Compute probabilities and focal weights
        probs = torch.sigmoid(logits)

        # p_t = probability of the TRUE class
        p_t = probs * targets_clean + (1.0 - probs) * (1.0 - targets_clean)

        # alpha_t = alpha weight for the TRUE class
        alpha_t = alpha * targets_clean + (1.0 - alpha) * (1.0 - targets_clean)

        # Standard BCE (unreduced, per-sample per-endpoint)
        bce = F.binary_cross_entropy_with_logits(logits, targets_clean, reduction='none')

        # Focal modulation: down-weight easy examples
        focal = alpha_t * (1.0 - p_t) ** self.gamma * bce

        # Apply mask and compute mean over known labels only
        masked_loss = focal * mask
        return masked_loss.sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def compute_pos_weights(Y_train: np.ndarray, target_cols: list) -> dict:
        """
        Utility: Compute per-endpoint positive class counts from training labels.
        Used to initialise PerEndpointFocalLoss.

        Args:
            Y_train     : (n, 12) label array with NaN for unknown.
            target_cols : Ordered list of endpoint names.

        Returns:
            Dict mapping endpoint → n_positive_examples.
        """
        pos_weights = {}
        for i, col in enumerate(target_cols):
            labels = Y_train[:, i]
            valid = labels[~np.isnan(labels)]
            n_pos = int((valid == 1).sum())
            n_neg = int((valid == 0).sum())
            # Weight = n_neg / n_pos (inverse frequency) — larger for rarer toxic class
            pos_weights[col] = n_neg / max(n_pos, 1)
        return pos_weights
