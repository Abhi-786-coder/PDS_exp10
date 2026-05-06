"""
src/stbi.py
Scaffold Toxicity Brittleness Index (STBI) — Original Framework.

What is STBI?
=============
For every Murcko scaffold in the training set that has BOTH toxic and non-toxic
members (for a given endpoint), STBI measures how structurally similar those
toxic and non-toxic members are to each other.

    STBI = 1 - min_Tanimoto_distance(toxic_set, safe_set)

STBI = 1.0  → Toxic and safe members of this scaffold differ by almost nothing.
               A single substituent change can flip toxicity. MAXIMUM BRITTLENESS.
               AI-guided substitution on this scaffold is HIGH RISK.

STBI = 0.0  → Toxic and safe members are structurally very different.
               The scaffold itself is not the primary driver of toxicity.
               Substitution is more reliable.

Why this matters:
-----------------
Every bioisostere suggestion system assumes the scaffold can be safely modified.
STBI makes the system self-aware: before generating suggestions, it checks
whether the scaffold even allows reliable modification. If STBI is high,
any suggestion — no matter how good — carries inherent uncertainty.

This concept does not appear in any published Tox21 bioisostere paper.
"""

import os
import pickle
import warnings
import numpy as np
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


# ─── Constants ────────────────────────────────────────────────────────────────

# STBI thresholds (calibrated by inspecting training set scaffold distributions)
STBI_EXTREME   = 0.80   # > 0.80: cliff-prone, suggestions unreliable
STBI_MODERATE  = 0.55   # 0.55–0.80: moderate risk, validate carefully
# < 0.55: robust scaffold, suggestions trustworthy


# ─── Core STBI Computation ─────────────────────────────────────────────────────

def _mol_to_fp(smiles: str, fp_radius: int = 2, fp_n_bits: int = 2048):
    """Convert SMILES to RDKit ExplicitBitVect. Returns None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, fp_n_bits)


def _get_scaffold(smiles: str) -> Optional[str]:
    """Return canonical Murcko scaffold SMILES, or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else None
    except Exception:
        return None


def compute_stbi_for_endpoint(
    df,
    endpoint:   str,
    fp_radius:  int = 2,
    fp_n_bits:  int = 2048,
    sample_cap: int = 30,
) -> dict:
    """
    Compute STBI for every Murcko scaffold for a single endpoint.

    Args:
        df:         Cleaned Tox21 DataFrame (must have 'smiles' column).
        endpoint:   Target column name (e.g., 'NR-AR').
        fp_radius:  Morgan fingerprint radius.
        fp_n_bits:  Fingerprint size (2048 is enough here — we're comparing,
                    not predicting, so 2048 is faster than 4096).
        sample_cap: Max molecules per scaffold class to use for speed.
                    30 is more than enough to find the minimum distance.

    Returns:
        Dict: {scaffold_smiles: stbi_score (float, 0.0–1.0)}
        Only scaffolds with BOTH toxic AND safe members are included.
    """
    col = df[endpoint]
    valid_mask = col.notna()
    df_valid = df[valid_mask].copy()

    # Group by scaffold
    scaffold_groups: dict[str, dict] = {}
    for _, row in df_valid.iterrows():
        smi   = row['smiles']
        label = row[endpoint]
        if label not in [0, 1]:
            continue
        scaffold = _get_scaffold(smi)
        if scaffold is None:
            continue
        if scaffold not in scaffold_groups:
            scaffold_groups[scaffold] = {'toxic': [], 'safe': []}
        key = 'toxic' if int(label) == 1 else 'safe'
        scaffold_groups[scaffold][key].append(smi)

    stbi_scores = {}
    for scaffold, members in scaffold_groups.items():
        toxic_smiles = members['toxic']
        safe_smiles  = members['safe']

        # Need at least 1 member on each side to compute brittleness
        if not toxic_smiles or not safe_smiles:
            continue

        # Cap sample size for speed
        if len(toxic_smiles) > sample_cap:
            rng = np.random.default_rng(42)
            toxic_smiles = list(rng.choice(toxic_smiles, size=sample_cap, replace=False))
        if len(safe_smiles) > sample_cap:
            rng = np.random.default_rng(42)
            safe_smiles = list(rng.choice(safe_smiles, size=sample_cap, replace=False))

        fps_toxic = [_mol_to_fp(s, fp_radius, fp_n_bits) for s in toxic_smiles]
        fps_safe  = [_mol_to_fp(s, fp_radius, fp_n_bits) for s in safe_smiles]
        fps_toxic = [f for f in fps_toxic if f is not None]
        fps_safe  = [f for f in fps_safe  if f is not None]

        if not fps_toxic or not fps_safe:
            continue

        # Find the MAXIMUM Tanimoto similarity between any toxic-safe pair
        # (= minimum distance pair = the closest cliff)
        max_cross_sim = 0.0
        for ft in fps_toxic:
            sims = DataStructs.BulkTanimotoSimilarity(ft, fps_safe)
            local_max = max(sims)
            if local_max > max_cross_sim:
                max_cross_sim = local_max
            if max_cross_sim > 0.98:
                break  # Early exit — already essentially identical

        # STBI = max similarity between toxic and safe members
        # High similarity = brittle (they look the same but behave differently)
        stbi_scores[scaffold] = round(float(max_cross_sim), 4)

    return stbi_scores


def compute_stbi_all_endpoints(df, target_cols: list, fp_radius: int = 2) -> dict:
    """
    Compute STBI across all endpoints.

    Returns:
        {
            scaffold_smiles: {
                endpoint: stbi_score,
                'max_stbi': float,       # worst-case endpoint
                'mean_stbi': float,      # average across endpoints with data
            }
        }
    """
    print("Computing Scaffold Toxicity Brittleness Index...")
    per_endpoint = {}
    for ep in target_cols:
        print(f"  Processing endpoint: {ep}...", end=' ', flush=True)
        scores = compute_stbi_for_endpoint(df, ep, fp_radius=fp_radius)
        per_endpoint[ep] = scores
        print(f"{len(scores)} scaffolds with cliff data.")

    # Aggregate: for each scaffold, collect scores across endpoints
    all_scaffolds = set()
    for ep_scores in per_endpoint.values():
        all_scaffolds.update(ep_scores.keys())

    aggregated = {}
    for scaffold in all_scaffolds:
        ep_scores = {}
        for ep in target_cols:
            if scaffold in per_endpoint[ep]:
                ep_scores[ep] = per_endpoint[ep][scaffold]
        if not ep_scores:
            continue
        vals = list(ep_scores.values())
        aggregated[scaffold] = {
            **ep_scores,
            'max_stbi':  round(max(vals), 4),
            'mean_stbi': round(float(np.mean(vals)), 4),
        }

    print(f"STBI computed for {len(aggregated)} scaffolds.")
    return aggregated


# ─── Lookup at Inference Time ──────────────────────────────────────────────────

def lookup_stbi(smiles: str, stbi_artifact: dict) -> dict:
    """
    Given a query molecule, find its Murcko scaffold and look up STBI.

    Args:
        smiles:         Input molecule SMILES.
        stbi_artifact:  Pre-computed STBI dict from compute_stbi_all_endpoints().

    Returns:
        {
            'scaffold':     str,
            'stbi':         float | None,
            'assessment':   'ROBUST' | 'MODERATE' | 'EXTREME' | 'UNSEEN',
            'risk_level':   int (0=robust, 1=moderate, 2=extreme, 3=unseen),
            'message':      str,
            'endpoint_scores': dict,
        }
    """
    scaffold = _get_scaffold(smiles)

    if scaffold is None:
        return {
            'scaffold':       None,
            'stbi':           None,
            'assessment':     'UNSEEN',
            'risk_level':     3,
            'message':        'Scaffold could not be extracted from this molecule.',
            'endpoint_scores': {},
        }

    entry = stbi_artifact.get(scaffold)

    if entry is None:
        return {
            'scaffold':       scaffold,
            'stbi':           None,
            'assessment':     'UNSEEN',
            'risk_level':     3,
            'message': (
                'This scaffold was not seen in training data. '
                'The model is operating outside its applicability domain. '
                'All predictions carry elevated uncertainty.'
            ),
            'endpoint_scores': {},
        }

    max_stbi = entry.get('max_stbi', 0.0)
    ep_scores = {k: v for k, v in entry.items() if k not in ('max_stbi', 'mean_stbi')}

    if max_stbi >= STBI_EXTREME:
        assessment = 'EXTREME'
        risk_level = 2
        message = (
            f'This scaffold is toxicity-brittle (STBI={max_stbi:.2f}). '
            'Molecules sharing this scaffold that are toxic and safe '
            'differ structurally by very little — small substituent changes '
            'can flip toxicity unpredictably. '
            'Any suggested replacement carries HIGH UNCERTAINTY. '
            'Experimental validation is mandatory.'
        )
    elif max_stbi >= STBI_MODERATE:
        assessment = 'MODERATE'
        risk_level = 1
        message = (
            f'This scaffold has moderate brittleness (STBI={max_stbi:.2f}). '
            'Some toxic and safe analogs are structurally similar. '
            'Suggestions are worth exploring but should be validated.'
        )
    else:
        assessment = 'ROBUST'
        risk_level = 0
        message = (
            f'This scaffold is toxicity-robust (STBI={max_stbi:.2f}). '
            'Toxic and safe members of this scaffold are structurally '
            'well-separated. Suggestions are more reliably guided.'
        )

    return {
        'scaffold':        scaffold,
        'stbi':            max_stbi,
        'mean_stbi':       entry.get('mean_stbi'),
        'assessment':      assessment,
        'risk_level':      risk_level,
        'message':         message,
        'endpoint_scores': ep_scores,
    }
