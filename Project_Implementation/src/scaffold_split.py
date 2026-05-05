"""
src/scaffold_split.py
Phase 2: 4-Way Murcko Scaffold Stratified Split.

Implements the mandatory 4-way scaffold split from the roadmap:
  - Train      : 70%
  - Validation : 10%
  - Calibration: 10%  ← Required for Mondrian Conformal Prediction (Phase 5)
  - Test       : 10%

ROADMAP WARNING: Random splitting on molecular data causes structural leakage —
similar scaffolds appear in both train and test, inflating AUC falsely.
This module uses Murcko scaffold split to ensure structural independence
between all four splits. (Zhang et al. 2025, Barua et al.)
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from typing import Tuple


def get_scaffold(smiles: str) -> str:
    """Extract Murcko scaffold SMILES from a molecule SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # fallback: treat full SMILES as its own scaffold
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_stratified_split(
    smiles_list: list,
    train_size: float = 0.70,
    val_size: float = 0.10,
    calib_size: float = 0.10,
    # test gets the remaining fraction (0.10)
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs a 4-way Murcko scaffold stratified split.

    Returns:
        (train_idx, val_idx, calib_idx, test_idx) — numpy integer index arrays
        Guaranteed to be mutually exclusive and collectively exhaustive.

    Args:
        smiles_list : List of SMILES strings (must be cleaned/canonical).
        train_size  : Fraction for training (default 0.70).
        val_size    : Fraction for validation (default 0.10).
        calib_size  : Fraction for Mondrian CP calibration (default 0.10).
        seed        : Random seed for reproducibility.
    """
    assert abs(train_size + val_size + calib_size - 0.90) < 1e-6, \
        "train_size + val_size + calib_size must equal 0.90 (test gets the rest)"

    # ── Step 1: Group molecule indices by scaffold ─────────────────────────────
    scaffold_to_idx = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        scaffold_to_idx[scaffold].append(i)

    # ── Step 2: Shuffle scaffolds (not molecules) ──────────────────────────────
    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_to_idx.keys())
    rng.shuffle(scaffolds)

    # ── Step 3: Fill buckets by molecule count ─────────────────────────────────
    n_total = len(smiles_list)
    n_train = int(np.floor(train_size * n_total))
    n_val   = int(np.floor(val_size   * n_total))
    n_calib = int(np.floor(calib_size * n_total))

    train_idx, val_idx, calib_idx, test_idx = [], [], [], []

    for scaffold in scaffolds:
        indices = scaffold_to_idx[scaffold]
        if len(train_idx) < n_train:
            train_idx.extend(indices)
        elif len(val_idx) < n_val:
            val_idx.extend(indices)
        elif len(calib_idx) < n_calib:
            calib_idx.extend(indices)
        else:
            test_idx.extend(indices)

    train_idx = np.array(train_idx, dtype=int)
    val_idx   = np.array(val_idx,   dtype=int)
    calib_idx = np.array(calib_idx, dtype=int)
    test_idx  = np.array(test_idx,  dtype=int)

    # ── Step 4: Sanity checks ──────────────────────────────────────────────────
    all_idx = np.concatenate([train_idx, val_idx, calib_idx, test_idx])
    assert len(all_idx) == n_total,          "Split indices do not cover all molecules!"
    assert len(set(all_idx)) == n_total,     "Duplicate indices found across splits!"

    print(f"✅ Scaffold split complete:")
    print(f"   Train:       {len(train_idx):5d}  ({len(train_idx)/n_total:.1%})")
    print(f"   Validation:  {len(val_idx):5d}  ({len(val_idx)/n_total:.1%})")
    print(f"   Calibration: {len(calib_idx):5d}  ({len(calib_idx)/n_total:.1%})")
    print(f"   Test:        {len(test_idx):5d}  ({len(test_idx)/n_total:.1%})")

    return train_idx, val_idx, calib_idx, test_idx
