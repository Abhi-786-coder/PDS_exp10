"""
src/pareto.py
Phase 6 — Step 4: Pareto dominance ranking of bioisostere candidates.

What is Pareto dominance here?
================================
Candidate A Pareto-dominates candidate B if:
  1. A has LOWER toxicity probability on EVERY endpoint than B, AND
  2. A has LOWER SCScore and SAScore than B (equal allowed), AND
  3. A is strictly better on at least one objective.

Candidates are first grouped into Pareto fronts (front 1 = non-dominated).
Within each front we use a weighted objective as a tie-breaker:

Weighted objective (lower = better):
  score = mean(toxicity_probs) + 0.1 * sc_score + 0.05 * sa_score

The weights give 90% importance to toxicity reduction and 10% to synthesizability.

OOD Detection
=============
Tanimoto similarity to the training set fingerprints.
  max_sim < 0.4 → OUT-OF-DISTRIBUTION warning (model unreliable)
  max_sim ≥ 0.4 → In-distribution (model trustworthy)

This is the roadmap-specified coverage check using training set proximity.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, asdict

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# ─── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class ParetoCandidate:
    """Single evaluated bioisostere candidate."""
    chembl_id:        str
    smiles:           str
    toxicity_probs:   dict          # {endpoint: probability}
    toxicity_flags:   dict          # {endpoint: bool (True = toxic)}
    mean_tox_prob:    float         # Mean across all 12 endpoints
    n_flagged:        int           # Number of toxic endpoints
    synth_verdict:    str           # HIGH_CONFIDENCE / SYNTHESIZABLE / REJECT
    sa_score:         float
    sc_score:         float
    adme:             dict          # delta_logp, delta_mw, passes
    ood_max_sim:      float         # Max Tanimoto to training set
    ood_warning:      bool          # True if max_sim < 0.4
    pareto_score:     float         # Weighted composite (lower = better)
    pareto_front:     int = 0       # Non-dominated sorting front (1 = best)
    pareto_status:    str = 'UNRANKED'
    rank:             int = 0       # Assigned after sorting

    def to_dict(self):
        return asdict(self)


# ─── OOD Detection ─────────────────────────────────────────────────────────────

def compute_ood_similarity(smiles_cand: str,
                            X_fp_train: np.ndarray,
                            radius: int = 2,
                            n_bits: int = 4096,
                            sample_size: int = 500) -> float:
    """
    Compute max Tanimoto similarity between a candidate molecule and the
    training set fingerprints.

    For efficiency we sample `sample_size` training molecules (not all 5460).
    Max-similarity is a conservative estimate — if the sampled maximum is < 0.4,
    the full set is very unlikely to have a higher similarity.

    Returns:
        max_sim (float): Maximum Tanimoto similarity to any training molecule.
    """
    mol = Chem.MolFromSmiles(smiles_cand)
    if mol is None:
        return 0.0

    fp_cand = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

    # Sample from training set for speed
    rng = np.random.default_rng(seed=42)
    n_train = len(X_fp_train)
    idx = rng.choice(n_train, size=min(sample_size, n_train), replace=False)

    max_sim = 0.0
    for i in idx:
        # Reconstruct RDKit ExplicitBitVect from numpy array
        bits = X_fp_train[i].astype(bool)
        fp_train = DataStructs.ExplicitBitVect(n_bits)
        for j in np.where(bits)[0]:
            fp_train.SetBit(int(j))
        sim = DataStructs.TanimotoSimilarity(fp_cand, fp_train)
        if sim > max_sim:
            max_sim = sim
        if max_sim > 0.9:   # Early exit — already very similar
            break

    return round(float(max_sim), 4)


# ─── Pareto Scoring ────────────────────────────────────────────────────────────

def compute_pareto_score(toxicity_probs: dict,
                          sc_score: float,
                          sa_score: float,
                          w_tox: float = 0.90,
                          w_sc:  float = 0.07,
                          w_sa:  float = 0.03) -> float:
    """
    Weighted composite score used as a tie-breaker within Pareto fronts.

    Weights:
      w_tox = 0.90: Toxicity reduction is the primary objective.
      w_sc  = 0.07: SCScore (normalized to 0–1 from 1–5 scale).
      w_sa  = 0.03: SAScore (normalized to 0–1 from 1–10 scale).
    """
    mean_tox = float(np.mean(list(toxicity_probs.values())))
    sc_norm  = (sc_score - 1.0) / 4.0     # SCScore: 1–5 → 0–1
    sa_norm  = (sa_score - 1.0) / 9.0     # SAScore: 1–10 → 0–1
    sc_norm  = float(np.clip(sc_norm, 0.0, 1.0))
    sa_norm  = float(np.clip(sa_norm, 0.0, 1.0))

    return round(w_tox * mean_tox + w_sc * sc_norm + w_sa * sa_norm, 6)


def pareto_dominates(a: ParetoCandidate, b: ParetoCandidate, eps: float = 1e-9) -> bool:
    """
    True if candidate `a` Pareto-dominates candidate `b`.

    Objectives (all minimized):
      - 12 endpoint toxicity probabilities
      - SCScore
      - SAScore
    """
    if not a.toxicity_probs or not b.toxicity_probs:
        return False

    keys_a = set(a.toxicity_probs.keys())
    keys_b = set(b.toxicity_probs.keys())
    if keys_a != keys_b:
        return False
    ordered_keys = sorted(keys_a)

    tox_a = np.array([float(a.toxicity_probs[k]) for k in ordered_keys], dtype=float)
    tox_b = np.array([float(b.toxicity_probs[k]) for k in ordered_keys], dtype=float)

    obj_a = np.concatenate((tox_a, np.array([float(a.sc_score), float(a.sa_score)], dtype=float)))
    obj_b = np.concatenate((tox_b, np.array([float(b.sc_score), float(b.sa_score)], dtype=float)))

    not_worse = np.all(obj_a <= (obj_b + eps))
    strictly_better = np.any(obj_a < (obj_b - eps))
    return bool(not_worse and strictly_better)


def _assign_pareto_fronts(candidates: list[ParetoCandidate]) -> None:
    """
    Assign non-dominated sorting fronts in-place (NSGA-style).
    Front 1 contains all non-dominated points.
    """
    n = len(candidates)
    dominates: list[set[int]] = [set() for _ in range(n)]
    dominated_count = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            if pareto_dominates(candidates[i], candidates[j]):
                dominates[i].add(j)
                dominated_count[j] += 1
            elif pareto_dominates(candidates[j], candidates[i]):
                dominates[j].add(i)
                dominated_count[i] += 1

    current_front = [idx for idx, count in enumerate(dominated_count) if count == 0]
    front_id = 1

    while current_front:
        next_front_set = set()
        for idx in current_front:
            candidates[idx].pareto_front = front_id
            candidates[idx].pareto_status = 'NON_DOMINATED' if front_id == 1 else 'DOMINATED'
            for dominated_idx in dominates[idx]:
                dominated_count[dominated_idx] -= 1
                if dominated_count[dominated_idx] == 0:
                    next_front_set.add(dominated_idx)
        current_front = sorted(next_front_set)
        front_id += 1


def rank_pareto_candidates(candidates: list[ParetoCandidate]) -> list[ParetoCandidate]:
    """
    Rank a list of ParetoCandidate by Pareto front, then weighted tie-breakers.

    Primary ordering:
      1. pareto_front ascending (1 is best / non-dominated)
      2. pareto_score ascending
      3. mean_tox_prob ascending
      4. n_flagged ascending
      5. sc_score ascending
      6. sa_score ascending

    Returns:
        Sorted list with .rank assigned starting from 1.
    """
    if not candidates:
        return []

    _assign_pareto_fronts(candidates)
    sorted_cands = sorted(
        candidates,
        key=lambda c: (
            c.pareto_front,
            c.pareto_score,
            c.mean_tox_prob,
            c.n_flagged,
            c.sc_score,
            c.sa_score,
        ),
    )
    for i, c in enumerate(sorted_cands):
        c.rank = i + 1
    return sorted_cands


# ─── Full Evaluation of a Single Candidate ─────────────────────────────────────

def evaluate_candidate(smiles_cand:    str,
                        chembl_id:     str,
                        smiles_orig:   str,
                        pipeline,                 # ToxNetPipeline
                        thresholds:    dict,
                        target_cols:   list,
                        synth_result:  dict,
                        adme_result:   dict,
                        X_fp_train:    np.ndarray,
                        device,
                        fp_radius:     int = 2,
                        fp_n_bits:     int = 4096) -> Optional[ParetoCandidate]:
    """
    Full evaluation of a single bioisostere candidate:
      1. Featurize (ECFP8, 4096 bits)
      2. Predict all 12 toxicity endpoints
      3. Compute OOD similarity
      4. Compute pareto_score

    Returns:
        ParetoCandidate, or None if featurization fails.
    """
    import torch

    # Featurize
    mol = Chem.MolFromSmiles(smiles_cand)
    if mol is None:
        return None

    fp_arr = np.zeros(fp_n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_n_bits)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    # Predict
    x = torch.from_numpy(fp_arr).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = pipeline(x).cpu().numpy()[0]   # (12,)

    tox_probs  = {col: round(float(probs[i]), 4) for i, col in enumerate(target_cols)}
    tox_flags  = {col: bool(probs[i] >= thresholds.get(col, 0.5))
                  for i, col in enumerate(target_cols)}
    n_flagged  = sum(tox_flags.values())

    # OOD
    ood_sim    = compute_ood_similarity(smiles_cand, X_fp_train, radius=fp_radius, n_bits=fp_n_bits)
    ood_warn   = ood_sim < 0.4

    # Pareto score
    pscore = compute_pareto_score(
        tox_probs,
        synth_result.get('sc_score', 3.0),
        synth_result.get('sa_score', 3.0),
    )

    return ParetoCandidate(
        chembl_id      = chembl_id,
        smiles         = smiles_cand,
        toxicity_probs = tox_probs,
        toxicity_flags = tox_flags,
        mean_tox_prob  = round(float(np.mean(list(tox_probs.values()))), 4),
        n_flagged      = n_flagged,
        synth_verdict  = synth_result.get('verdict', 'UNKNOWN'),
        sa_score       = synth_result.get('sa_score', 99.0),
        sc_score       = synth_result.get('sc_score', 99.0),
        adme           = adme_result,
        ood_max_sim    = ood_sim,
        ood_warning    = ood_warn,
        pareto_score   = pscore,
    )
