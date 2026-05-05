"""
src/geometric_imbalance.py
Phase 3A: Intraclass Tanimoto Cohesion Analysis — Novel Research Finding.

Measures whether the toxic class is geometrically clustered in fingerprint space
per endpoint. A high cohesion ratio means toxic compounds are structurally
similar to each other but different from the safe class.

Key finding this module produces:
  - High cohesion ratio → SMOTE is INVALID (interpolating within a tight cluster
    produces near-identical synthetic points, adding no new information).
  - Low cohesion ratio → SMOTE is VALID (toxic class is diffuse; interpolation
    explores real chemical space between genuinely diverse toxic compounds).

This predicts where SMOTE-augmented training will help or hurt — a finding
no prior Tox21 paper has quantified (Zhang et al. 2025, Barua et al.).
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
import warnings


def _get_fps(smiles_list: list, radius: int = 2, n_bits: int = 4096) -> list:
    """Compute Morgan fingerprint objects (not arrays) for Tanimoto similarity."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))
        else:
            fps.append(None)
    return fps


def _mean_pairwise_tanimoto(fp_objects: list, sample_size: int = 200, rng=None) -> float:
    """
    Compute mean pairwise Tanimoto similarity within a set of molecules.
    Subsampled to `sample_size` for computational tractability.
    """
    fp_valid = [fp for fp in fp_objects if fp is not None]
    if len(fp_valid) < 2:
        return 0.0

    if rng is None:
        rng = np.random.default_rng(42)

    if len(fp_valid) > sample_size:
        idx = rng.choice(len(fp_valid), sample_size, replace=False)
        fp_valid = [fp_valid[i] for i in idx]

    sims = [
        DataStructs.TanimotoSimilarity(fp_valid[i], fp_valid[j])
        for i in range(len(fp_valid))
        for j in range(i + 1, len(fp_valid))
    ]
    return float(np.mean(sims)) if sims else 0.0


def compute_cohesion_table(
    smiles_list: list,
    Y: np.ndarray,
    target_cols: list,
    sample_size: int = 200,
    radius: int = 2,
    n_bits: int = 4096,
) -> pd.DataFrame:
    """
    Compute intraclass Tanimoto cohesion for each of the 12 Tox21 endpoints.

    Args:
        smiles_list : List of SMILES (training set only — never expose test set).
        Y           : Label array (n_molecules, 12). NaN = unknown.
        target_cols : List of endpoint column names (length 12).
        sample_size : Max molecules to use per class for pairwise calculation.
        radius / n_bits : Morgan fingerprint parameters.

    Returns:
        DataFrame with columns:
            endpoint, n_toxic, n_safe, toxic_cohesion,
            nontoxic_cohesion, cohesion_ratio
    """
    rng = np.random.default_rng(42)
    # Precompute all fingerprints once
    all_fps = _get_fps(smiles_list, radius=radius, n_bits=n_bits)

    rows = []
    for i, col in enumerate(target_cols):
        labels = Y[:, i]
        toxic_mask    = labels == 1.0
        nontoxic_mask = labels == 0.0

        toxic_fps    = [all_fps[j] for j in np.where(toxic_mask)[0]]
        nontoxic_fps = [all_fps[j] for j in np.where(nontoxic_mask)[0]]

        toxic_coh    = _mean_pairwise_tanimoto(toxic_fps,    sample_size, rng)
        nontoxic_coh = _mean_pairwise_tanimoto(nontoxic_fps, sample_size, rng)
        ratio        = toxic_coh / max(nontoxic_coh, 1e-6)

        rows.append({
            'endpoint':          col,
            'n_toxic':           int(toxic_mask.sum()),
            'n_safe':            int(nontoxic_mask.sum()),
            'toxic_cohesion':    round(toxic_coh,    4),
            'nontoxic_cohesion': round(nontoxic_coh, 4),
            'cohesion_ratio':    round(ratio,         3),
        })

    return pd.DataFrame(rows).sort_values('cohesion_ratio', ascending=False)


def find_optimal_smote_threshold(
    cohesion_df: pd.DataFrame,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    target_cols: list,
    candidate_thresholds: list = None,
) -> dict:
    """
    Empirically find the optimal cohesion_ratio threshold that maximises
    Macro AUPRC on the validation set.

    ROADMAP: We do NOT hardcode 1.5. The data picks the threshold.
    A molecule endpoint is SMOTE-valid if its cohesion_ratio < threshold.

    Returns dict with keys: threshold, macro_auprc, results_table
    """
    if candidate_thresholds is None:
        candidate_thresholds = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]

    cohesion_lookup = dict(zip(cohesion_df['endpoint'], cohesion_df['cohesion_ratio']))
    threshold_results = {}

    for t in candidate_thresholds:
        smote_map = {ep: cohesion_lookup.get(ep, 0.0) < t for ep in target_cols}
        auprc = _eval_with_smote_map(smote_map, X_train, Y_train, X_val, Y_val, target_cols)
        threshold_results[t] = round(auprc, 4)
        print(f"  Threshold {t}: Macro AUPRC = {auprc:.4f}")

    optimal = max(threshold_results, key=threshold_results.get)
    print(f"\n✅ Optimal cohesion threshold: {optimal} (AUPRC = {threshold_results[optimal]:.4f})")

    return {
        'threshold':   optimal,
        'macro_auprc': threshold_results[optimal],
        'all_results': threshold_results,
    }


def _eval_with_smote_map(
    smote_map: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    target_cols: list,
) -> float:
    """
    Train a lightweight LR per endpoint (with or without SMOTE depending on
    the smote_map) and return Macro AUPRC on the validation set.
    Used only during the sensitivity analysis — not for final model training.
    """
    auprc_scores = []

    for i, col in enumerate(target_cols):
        y_tr = Y_train[:, i]
        y_v  = Y_val[:, i]
        tm = ~np.isnan(y_tr)
        vm = ~np.isnan(y_v)

        if tm.sum() < 10 or vm.sum() < 5:
            continue
        if len(np.unique(y_tr[tm])) < 2 or len(np.unique(y_v[vm])) < 2:
            continue

        X_tr_ep = X_train[tm]
        y_tr_ep = y_tr[tm]

        if smote_map.get(col, False):
            try:
                n_minority = int((y_tr_ep == 1).sum())
                if n_minority >= 2:
                    k = min(5, n_minority - 1)
                    sm = SMOTE(random_state=42, k_neighbors=k)
                    X_tr_ep, y_tr_ep = sm.fit_resample(X_tr_ep, y_tr_ep)
            except Exception:
                pass  # Fall back to no SMOTE if it fails

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=300, solver='lbfgs', class_weight='balanced')
            clf.fit(X_tr_ep, y_tr_ep)

        y_prob = clf.predict_proba(X_val[vm])[:, 1]
        auprc  = average_precision_score(y_v[vm], y_prob)
        auprc_scores.append(auprc)

    return float(np.mean(auprc_scores)) if auprc_scores else 0.0
