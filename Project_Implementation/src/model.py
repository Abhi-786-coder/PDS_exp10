"""
src/model.py
Phase 4: ToxNet Architecture — Shared Backbone + 12 Per-Endpoint Heads.

Design rationale:
  - Shared backbone forces the network to learn generalised molecular representations.
    All 12 endpoints see the same molecules, so shared chemistry knowledge transfers.
  - Per-endpoint heads specialise independently. This is critical because endpoint
    difficulty, class balance, and chemical space coverage differ dramatically across
    the 12 Tox21 targets (e.g., NR-AR-LBD is 2.8% toxic; SR-MMP is 16.9% toxic).
  - BatchNorm1d after each Linear layer stabilises training on sparse binary inputs
    (Morgan fingerprints are ~98% zeros for typical molecular bit vectors).
  - GELU activation is preferred over ReLU for molecular property prediction tasks
    (smoother gradient flow through sparse inputs).

Two model sizes are defined:
  - ToxNetLite: Fast, used only in Pass 1 to generate continuous embeddings.
    Input: 4096-bit fingerprints → [512, 256] backbone → 12 heads
    Purpose: feature compression for SMOTE-valid endpoints.

  - ToxNet: Full model, trained in Pass 2 on the 256-dim continuous embeddings.
    Input: 256-dim embeddings → [256, 128] backbone → 12 heads
    Purpose: final toxicity prediction with augmented training data.
"""

import torch
import torch.nn as nn


class ToxNetLite(nn.Module):
    """
    Pass 1 model. Compresses 4096-bit fingerprints to 256-dim continuous embeddings.
    Trains fast (20 epochs) with Focal Loss. No SMOTE. Exists only to produce
    chemically meaningful continuous embeddings for the SMOTE augmentation step.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        shared_dims: list = None,
        head_dim: int = 32,
        n_tasks: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()
        if shared_dims is None:
            shared_dims = [512, 256]

        # Build shared backbone
        layers, prev = [], input_dim
        for h in shared_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.embedding_dim = prev  # 256 — the dim SMOTE will operate in

        # 12 lightweight per-endpoint heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(head_dim, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 12) raw logits."""
        shared = self.backbone(x)
        return torch.cat([h(shared) for h in self.heads], dim=1)

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 256) continuous embedding vectors for SMOTE."""
        self.eval()
        return self.backbone(x)


class ToxNet(nn.Module):
    """
    Pass 2 model. Final toxicity predictor trained on SMOTE-augmented 256-dim embeddings.
    This is the model used for evaluation, SHAP attribution (Phase 5), and
    the prescription pipeline (Phase 6).

    Input: 256-dim continuous embeddings from ToxNetLite backbone.
    """

    def __init__(
        self,
        input_dim: int = 256,
        shared_dims: list = None,
        head_dim: int = 64,
        n_tasks: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()
        if shared_dims is None:
            shared_dims = [256, 128]

        # Build shared backbone
        layers, prev = [], input_dim
        for h in shared_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.embedding_dim = prev  # 128

        # 12 independent per-endpoint heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(head_dim, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 12) raw logits."""
        shared = self.backbone(x)
        return torch.cat([h(shared) for h in self.heads], dim=1)

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 128) embeddings — used for SHAP DeepExplainer in Phase 6."""
        self.eval()
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 12) sigmoid probabilities (0-1 range)."""
        return torch.sigmoid(self.forward(x))
