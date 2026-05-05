"""
src/dataset.py
PyTorch Dataset for multi-task Tox21 molecular toxicity prediction.

Critical design decisions:
  - NaN labels (unknown experiments) are converted to -1.0 sentinel INSIDE __getitem__.
    The PerEndpointFocalLoss masks these out during backward pass.
    This ensures the model NEVER propagates gradient from unknown experimental data.
  - Fingerprints are stored as float32 (not uint8) because PyTorch Linear layers
    require float inputs. Conversion happens once at dataset init, not per-batch.
  - A custom collate function is NOT needed — standard torch.stack handles this.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ToxDataset(Dataset):
    """
    Multi-task toxicity dataset. Handles NaN → -1 sentinel conversion.

    Args:
        X       : (n, features) numpy float32 feature matrix.
        Y       : (n, 12) numpy float array. NaN = label not available.
        device  : 'cuda' or 'cpu'. Tensors pre-moved for speed.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))

        # Convert NaN → -1.0 sentinel. -1 is masked in PerEndpointFocalLoss.
        Y_sentinel = Y.copy().astype(np.float32)
        Y_sentinel[np.isnan(Y_sentinel)] = -1.0
        self.Y = torch.from_numpy(Y_sentinel)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class EmbeddingDataset(Dataset):
    """
    Dataset for Pass 2 training on 256-dim continuous embeddings.
    Used after ToxNetLite has extracted embeddings and SMOTE has augmented them.

    For SMOTE-valid endpoints, Y may have fractional interpolated labels (0.0 or 1.0
    from ADASYN — ADASYN always outputs clean binary labels, not fractions).
    """

    def __init__(self, embeddings: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(embeddings.astype(np.float32))
        Y_sentinel = Y.copy().astype(np.float32)
        Y_sentinel[np.isnan(Y_sentinel)] = -1.0
        self.Y = torch.from_numpy(Y_sentinel)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    use_embedding: bool = False,
) -> DataLoader:
    """
    Convenience factory for creating DataLoaders.

    Args:
        X             : Feature matrix (fingerprints or embeddings).
        Y             : Label matrix (NaN for unknown).
        batch_size    : Mini-batch size.
        shuffle       : True for training, False for val/test.
        use_embedding : If True, uses EmbeddingDataset (for Pass 2 training).
    """
    dataset_cls = EmbeddingDataset if use_embedding else ToxDataset
    ds = dataset_cls(X, Y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,       # 0 workers inside Docker — avoids multiprocessing issues
        pin_memory=True,     # Faster GPU transfer when using CUDA
        drop_last=False,
    )
