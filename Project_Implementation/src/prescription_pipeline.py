"""
src/prescription_pipeline.py
Phase 6 — Master orchestrator: SMILES → Pareto-ranked bioisostere candidates.

4-Step Corrected Flow (from roadmap):
  Step 1:  Predict all 12 endpoints + compute SHAP per endpoint
  Step 1b: Cross-validate SHAP against structural alert databases (Brenk/PAINS)
           Flag low-confidence attributions
  Step 2:  Map top SHAP bit → molecular fragment (RDKit bitInfo)
  Step 3:  Query ChEMBL cache for molecules containing replacement fragments
           → Filter A: Synthesizability consensus (SAScore + SCScore)
           → Filter B: ADME preservation (|ΔLogP| < 0.5, |ΔMW| < 25 Da)
  Step 4:  Re-predict all 12 endpoints for each viable candidate
           → OOD detection (Tanimoto to training set)
           → Pareto dominance ranking
           → Output sorted Pareto front

Architectural Notes
===================
- The pipeline is STATELESS: it receives all dependencies as arguments.
  This makes it testable, composable, and easy to wire into FastAPI.
- ChEMBL is queried from a pre-built LOCAL CACHE only (no live API calls).
  If the cache does not contain a fragment, we fall back to a set of
  hand-curated PAINS-replacement SMILES that are always available.
- SHAP values are re-computed on the single input molecule (single-molecule
  GradientExplainer) — we do NOT use the pre-computed batch SHAP values
  from Phase 5 because those are for the calibration set, not new inputs.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import torch
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, FilterCatalog, DataStructs

from src.bioisostere import (
    synthesizability_consensus,
    adme_delta,
    extract_fragment_from_bit,
    load_chembl_cache,
    query_chembl_cache,
)
from src.pareto import evaluate_candidate, rank_pareto_candidates


# ─── PAINS / Brenk Alert Filtering ────────────────────────────────────────────

def _build_alert_catalogs():
    """Build RDKit PAINS + Brenk structural alert catalogs."""
    params_pains = FilterCatalog.FilterCatalogParams()
    params_pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params_pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params_pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)

    params_brenk = FilterCatalog.FilterCatalogParams()
    params_brenk.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)

    return (
        FilterCatalog.FilterCatalog(params_pains),
        FilterCatalog.FilterCatalog(params_brenk),
    )


_PAINS_CATALOG, _BRENK_CATALOG = _build_alert_catalogs()


def check_structural_alerts(smiles: str) -> dict:
    """
    Cross-validate a SMILES against PAINS and Brenk alert databases.

    Returns:
        {
            'pains_hit':  bool,
            'brenk_hit':  bool,
            'alert_names': list[str],
            'shap_confidence': 'HIGH' | 'MEDIUM' | 'LOW'
        }

    SHAP confidence logic (roadmap Step 1b):
      - No alerts: HIGH (SHAP attribution is likely reliable)
      - 1 alert:   MEDIUM (SHAP may be attributing to a known reactive group)
      - 2+ alerts: LOW (SHAP attribution should be treated with caution)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'pains_hit': False, 'brenk_hit': False,
                'alert_names': [], 'shap_confidence': 'LOW'}

    pains_hit  = _PAINS_CATALOG.HasMatch(mol)
    brenk_hit  = _BRENK_CATALOG.HasMatch(mol)
    alert_names = []

    for entry in _PAINS_CATALOG.GetMatches(mol):
        alert_names.append(f'PAINS:{entry.GetDescription()}')
    for entry in _BRENK_CATALOG.GetMatches(mol):
        alert_names.append(f'Brenk:{entry.GetDescription()}')

    n_alerts = len(alert_names)
    if n_alerts == 0:
        shap_conf = 'HIGH'
    elif n_alerts == 1:
        shap_conf = 'MEDIUM'
    else:
        shap_conf = 'LOW'

    return {
        'pains_hit':        pains_hit,
        'brenk_hit':        brenk_hit,
        'alert_names':      alert_names,
        'shap_confidence':  shap_conf,
    }


# ─── Step 1: Predict + SHAP ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(smiles: str,
                   pipeline,
                   thresholds: dict,
                   target_cols: list,
                   device,
                   fp_radius: int = 2,
                   fp_n_bits: int = 4096) -> Optional[dict]:
    """
    Featurize + predict a single SMILES molecule.

    Returns:
        {
            'fp': np.ndarray (4096,),
            'probabilities': {endpoint: float},
            'flags': {endpoint: bool},
            'n_flagged': int,
            'mean_prob': float,
        }
        or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp_arr = np.zeros(fp_n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_n_bits)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    x = torch.from_numpy(fp_arr).unsqueeze(0).to(device)
    probs = pipeline(x).cpu().numpy()[0]   # (12,)

    probabilities = {col: round(float(probs[i]), 4) for i, col in enumerate(target_cols)}
    flags         = {col: bool(probs[i] >= thresholds.get(col, 0.5))
                     for i, col in enumerate(target_cols)}
    n_flagged     = sum(flags.values())

    return {
        'fp':            fp_arr,
        'probabilities': probabilities,
        'flags':         flags,
        'n_flagged':     n_flagged,
        'mean_prob':     round(float(np.mean(probs)), 4),
    }


def compute_shap_single(smiles: str,
                         fp_arr: np.ndarray,
                         pipeline,
                         X_train_bg: np.ndarray,
                         target_cols: list,
                         device,
                         n_background: int = 100) -> Optional[np.ndarray]:
    """
    Compute SHAP GradientExplainer values for a single molecule.

    Returns:
        shap_arr: (12, 4096) — SHAP value per endpoint per bit. None on failure.
    """
    try:
        import shap
        pipeline.eval()

        rng = np.random.default_rng(42)
        bg_idx = rng.choice(len(X_train_bg), size=min(n_background, len(X_train_bg)), replace=False)
        X_bg   = X_train_bg[bg_idx].astype(np.float32)

        bg_tensor  = torch.from_numpy(X_bg).to(device)
        exp_tensor = torch.from_numpy(fp_arr[None, :]).to(device)  # (1, 4096)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer  = shap.GradientExplainer(pipeline, bg_tensor)
            raw_shap   = explainer.shap_values(exp_tensor)  # (1, 4096, 12) or list

        if isinstance(raw_shap, list):
            # list of 12 arrays each (1, 4096) → stack to (1, 12, 4096) → squeeze
            shap_arr = np.stack(raw_shap, axis=1)[0]   # (12, 4096)
        else:
            shap_arr = raw_shap[0].T                    # (4096, 12) → (12, 4096)

        return shap_arr   # (12, 4096)

    except Exception as e:
        warnings.warn(f'SHAP computation failed: {e}')
        return None


# ─── Step 2: Map bits → fragments ─────────────────────────────────────────────

def get_top_toxic_bits(shap_arr: np.ndarray,
                        flagged_endpoints: list,
                        target_cols: list,
                        top_k: int = 5) -> list:
    """
    Get the top-k fingerprint bits driving toxicity across all flagged endpoints.

    Strategy: sum absolute SHAP values across flagged endpoints only.
    This focuses the bioisostere search on the bits causing flagged toxicities,
    not bits that happen to reduce safe endpoints.

    Returns:
        List of (bit_index, cumulative_importance) tuples, sorted descending.
    """
    if shap_arr is None or not flagged_endpoints:
        return []

    # Build mask for flagged endpoints
    flagged_idx = [target_cols.index(ep) for ep in flagged_endpoints
                   if ep in target_cols]
    if not flagged_idx:
        return []

    # Sum |SHAP| across flagged endpoints: (n_flagged, 4096) → (4096,)
    summed = np.abs(shap_arr[flagged_idx, :]).sum(axis=0)
    top_bits = np.argsort(summed)[::-1][:top_k]
    return [(int(b), float(summed[b])) for b in top_bits]


# ─── Step 3: Filter candidates ────────────────────────────────────────────────

def filter_candidates(candidates_raw: list,
                       smiles_orig: str) -> list:
    """
    Apply synthesizability + ADME filters to raw ChEMBL candidates.

    Returns:
        List of dicts: raw candidate + 'synth' + 'adme' keys added.
        Only candidates with synth.include=True AND adme.passes=True are kept.
    """
    filtered = []
    for cand in candidates_raw:
        smi = cand.get('smiles')
        if not smi or Chem.MolFromSmiles(smi) is None:
            continue
        if smi == smiles_orig:
            continue  # Skip the original molecule itself

        synth = synthesizability_consensus(smi)
        if not synth['include']:
            continue   # REJECT — too hard to synthesize

        adme = adme_delta(smiles_orig, smi)
        if not adme.get('passes', False):
            continue   # Destroys ADME properties

        filtered.append({**cand, 'synth': synth, 'adme': adme})

    return filtered


def _prioritize_candidates_by_similarity(smiles_orig: str,
                                         candidates: list,
                                         fp_radius: int = 2,
                                         fp_n_bits: int = 4096) -> list:
    """
    Rank candidate molecules by fingerprint similarity to the input molecule.
    Higher similarity candidates are more likely to preserve ADME and pass filters.
    """
    mol_orig = Chem.MolFromSmiles(smiles_orig)
    if mol_orig is None or not candidates:
        return candidates

    fp_orig = AllChem.GetMorganFingerprintAsBitVect(
        mol_orig, radius=fp_radius, nBits=fp_n_bits
    )

    scored = []
    for cand in candidates:
        smi = cand.get('smiles')
        mol_cand = Chem.MolFromSmiles(smi) if smi else None
        if mol_cand is None:
            continue
        fp_cand = AllChem.GetMorganFingerprintAsBitVect(
            mol_cand, radius=fp_radius, nBits=fp_n_bits
        )
        sim = float(DataStructs.TanimotoSimilarity(fp_orig, fp_cand))
        scored.append({**cand, '_sim_to_input': round(sim, 4)})

    scored.sort(key=lambda c: c.get('_sim_to_input', 0.0), reverse=True)
    return scored


def _generate_curated_fallback_candidates(smiles_orig: str) -> list:
    """
    Deterministic local fallback candidate generation.
    Applies conservative halogen swaps (Cl/Br/I -> F/O) to create close analogs
    when fragment-cache retrieval yields no viable candidates.
    """
    mol = Chem.MolFromSmiles(smiles_orig)
    if mol is None:
        return []

    fallback = []
    seen = set()
    swap_targets = {17: [9, 8], 35: [9, 8], 53: [9, 8]}  # Cl/Br/I -> F/O

    for atom_idx, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in swap_targets:
            continue
        for new_atomic_num in swap_targets[atomic_num]:
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(atom_idx).SetAtomicNum(new_atomic_num)
            try:
                cand_mol = rw.GetMol()
                Chem.SanitizeMol(cand_mol)
                cand_smiles = Chem.MolToSmiles(cand_mol, canonical=True)
            except Exception:
                continue
            if not cand_smiles or cand_smiles == smiles_orig or cand_smiles in seen:
                continue
            seen.add(cand_smiles)
            fallback.append({
                'chembl_id': f'CURATED_SWAP_{atomic_num}_TO_{new_atomic_num}',
                'smiles': cand_smiles,
                'source': 'curated_fallback',
            })

    return fallback


# ─── Master Pipeline ──────────────────────────────────────────────────────────

def run_prescription_pipeline(
    smiles:         str,
    pipeline,                  # ToxNetPipeline
    thresholds:     dict,
    target_cols:    list,
    X_train_bg:     np.ndarray,
    X_fp_train:     np.ndarray,
    device,
    chembl_cache:   dict,
    fp_radius:      int = 2,
    fp_n_bits:      int = 4096,
    top_shap_bits:  int = 3,
    max_candidates: int = 20,
    shap_bg_size:   int = 100,
) -> dict:
    """
    Full 4-step prescription pipeline.

    Args:
        smiles:          Input molecule SMILES.
        pipeline:        ToxNetPipeline (end-to-end model).
        thresholds:      Per-endpoint calibrated thresholds from Phase 5.
        target_cols:     List of 12 endpoint names.
        X_train_bg:      Training fingerprints (for SHAP background).
        X_fp_train:      Training fingerprints (for OOD check).
        device:          torch.device.
        chembl_cache:    Pre-built ChEMBL fragment cache (from load_chembl_cache).
        fp_radius:       Morgan radius (default=2 = ECFP4 environment).
        fp_n_bits:       FP size (default=4096).
        top_shap_bits:   How many top SHAP bits to use for bioisostere search.
        max_candidates:  Max candidates to evaluate in Step 4 (for speed).
        shap_bg_size:    Background sample size for SHAP.

    Returns:
        {
            'input_smiles':       str,
            'is_valid':           bool,
            'alerts':             dict  (PAINS/Brenk),
            'prediction':         dict  (probabilities, flags, n_flagged),
            'shap_top_bits':      list  (bit, importance),
            'pareto_candidates':  list  (ranked ParetoCandidate dicts),
            'n_candidates_found': int,
            'n_after_filter':     int,
            'pipeline_status':    str,
        }
    """
    result = {
        'input_smiles':      smiles,
        'is_valid':          False,
        'alerts':            {},
        'prediction':        {},
        'shap_top_bits':     [],
        'pareto_candidates': [],
        'n_candidates_found': 0,
        'n_after_filter':    0,
        'pipeline_status':   'pending',
    }

    # ── Validate SMILES ────────────────────────────────────────────────────────
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result['pipeline_status'] = 'error:invalid_smiles'
        return result
    result['is_valid'] = True

    # ── Step 1: Predict ────────────────────────────────────────────────────────
    pred = predict_single(smiles, pipeline, thresholds, target_cols, device,
                          fp_radius, fp_n_bits)
    if pred is None:
        result['pipeline_status'] = 'error:featurization_failed'
        return result
    result['prediction'] = {
        'probabilities': pred['probabilities'],
        'flags':         pred['flags'],
        'n_flagged':     pred['n_flagged'],
        'mean_prob':     pred['mean_prob'],
    }

    # ── Step 1b: Structural alerts ─────────────────────────────────────────────
    alerts = check_structural_alerts(smiles)
    result['alerts'] = alerts

    # If no endpoints flagged — molecule is clean, no bioisostere needed
    flagged_endpoints = [ep for ep, flag in pred['flags'].items() if flag]
    if not flagged_endpoints:
        result['pipeline_status'] = 'clean_no_action_needed'
        return result

    # ── Step 1 SHAP ────────────────────────────────────────────────────────────
    shap_arr = compute_shap_single(
        smiles, pred['fp'], pipeline, X_train_bg, target_cols, device, shap_bg_size
    )
    top_bits = get_top_toxic_bits(shap_arr, flagged_endpoints, target_cols, top_shap_bits)
    result['shap_top_bits'] = top_bits

    # ── Step 2: Map bits → fragments ───────────────────────────────────────────
    fragments_to_query = []
    for bit_idx, _ in top_bits:
        frag = extract_fragment_from_bit(smiles, bit_idx, radius=fp_radius, n_bits=fp_n_bits)
        if frag and frag not in fragments_to_query:
            fragments_to_query.append(frag)

    # ── Step 3: Query ChEMBL cache + filter ────────────────────────────────────
    raw_candidates = []
    for frag in fragments_to_query:
        hits = query_chembl_cache(chembl_cache, frag)
        raw_candidates.extend(hits)

    # Deduplicate by SMILES
    seen = set()
    deduped = []
    for c in raw_candidates:
        if c['smiles'] not in seen:
            seen.add(c['smiles'])
            deduped.append(c)

    result['n_candidates_found'] = len(deduped)

    # Prioritize close analogs before filtering. This avoids spending the
    # filter budget on distant cache hits that are very unlikely to pass ADME.
    prioritized = _prioritize_candidates_by_similarity(
        smiles, deduped, fp_radius=fp_radius, fp_n_bits=fp_n_bits
    )

    # Apply filters on a larger, similarity-sorted pool.
    filter_pool_size = max(max_candidates * 15, max_candidates)
    filtered = filter_candidates(prioritized[:filter_pool_size], smiles)

    # If still empty, broaden search depth while keeping strict filters.
    if not filtered and prioritized:
        broad_pool_size = min(len(prioritized), max(max_candidates * 40, filter_pool_size))
        filtered = filter_candidates(prioritized[:broad_pool_size], smiles)

    # Final deterministic fallback from local curated transformations.
    if not filtered:
        curated_raw = _generate_curated_fallback_candidates(smiles)
        if curated_raw:
            filtered = filter_candidates(curated_raw, smiles)

    result['n_after_filter'] = len(filtered)

    if not filtered:
        result['pipeline_status'] = 'no_viable_candidates'
        return result

    # ── Step 4: Re-predict + Pareto rank ───────────────────────────────────────
    evaluated = []
    for cand in filtered[:max_candidates]:
        pc = evaluate_candidate(
            smiles_cand  = cand['smiles'],
            chembl_id    = cand.get('chembl_id', 'UNKNOWN'),
            smiles_orig  = smiles,
            pipeline     = pipeline,
            thresholds   = thresholds,
            target_cols  = target_cols,
            synth_result = cand['synth'],
            adme_result  = cand['adme'],
            X_fp_train   = X_fp_train,
            device       = device,
            fp_radius    = fp_radius,
            fp_n_bits    = fp_n_bits,
        )
        if pc is not None:
            evaluated.append(pc)

    ranked = rank_pareto_candidates(evaluated)
    result['pareto_candidates'] = [c.to_dict() for c in ranked]
    result['pipeline_status'] = 'complete'
    return result
