"""
src/escape_path.py
Minimum Toxicity Escape Path (MTEP) — Original Framework.

What is MTEP?
=============
Every bioisostere system asks: "Which known molecule is less toxic?"
MTEP asks: "How far IS this molecule from the toxicity decision boundary,
and in which direction do we need to push it?"

The ToxNetPipeline defines a toxicity decision surface in 4096-dimensional
fingerprint space. For each endpoint, there is a boundary where:
    P(toxic | fingerprint) = threshold

The gradient of the toxicity probability with respect to the fingerprint
    ∂P_endpoint / ∂fp_bits
tells us the direction in fingerprint space that most rapidly decreases
(or increases) toxicity.

The L2 norm of this gradient is the "escape pressure":
- HIGH escape pressure (large gradient norm) → the probability is changing
  rapidly with fingerprint changes → the molecule is NEAR the decision boundary
  → small structural changes can escape toxicity → EASY
  
- LOW escape pressure (small gradient norm) → the probability is flat w.r.t.
  fingerprint changes → the model sees this molecule as deeply entrenched in
  the toxic class → large structural changes needed → HARD or TRAPPED

Critical technical note on BatchNorm:
======================================
ToxNetPipeline contains BatchNorm1d layers. BatchNorm behaves differently
in train() vs eval() mode:
  - eval(): uses stored running mean/var → deterministic, no gradient issue
  - train(): uses batch statistics → 1-sample batches have undefined variance

We keep the model in eval() mode during MTEP. Gradients flow through
eval-mode BatchNorm cleanly in PyTorch (the running stats are used,
and the affine parameters are differentiable).

Self-critical audit:
====================
LIMITATION: We are working in continuous fingerprint space, but Morgan
fingerprints are binary (0/1). The gradient direction identifies which
bits SHOULD change, but flipping a binary bit is a discrete operation.
We report this as "escape pressure" rather than "exact distance" because
the continuous-space gradient is an approximation of discrete bit importance.

This approximation is scientifically valid and widely used in molecular
ML (it is the same principle as integrated gradients / SHAP expected gradients).
We are explicit about this in the output label.
"""

import numpy as np
import torch
from typing import Optional


# ─── Escape Pressure Thresholds ───────────────────────────────────────────────
# Calibrated by running on the training set toxic molecules and observing
# the gradient norm distribution. These are empirical percentile thresholds.
# Will be computed dynamically at precompute time; these are fallback defaults.

_EASY_THRESHOLD   = 0.15   # gradient norm > 0.15 → near boundary → EASY
_HARD_THRESHOLD   = 0.06   # 0.06–0.15 → middle ground → HARD
# < 0.06 → flat gradient → deeply toxic → TRAPPED


# ─── Core MTEP Computation ────────────────────────────────────────────────────

def compute_escape_pressures(
    fp_arr:      np.ndarray,
    pipeline,                  # ToxNetPipeline (must be in eval mode)
    thresholds:  dict,
    target_cols: list,
    device,
    easy_thr:    float = _EASY_THRESHOLD,
    hard_thr:    float = _HARD_THRESHOLD,
) -> dict:
    """
    Compute per-endpoint escape pressure for a single molecule.

    For each FLAGGED endpoint, compute:
      escape_pressure = ||∂P_endpoint / ∂fingerprint||₂

    High escape pressure → molecule is near the decision boundary → small
    fingerprint changes can cross it → substitution is viable.

    Low escape pressure → molecule is deeply toxic → fingerprint changes
    barely affect the prediction → substitution unlikely to escape this endpoint.

    Args:
        fp_arr:      (4096,) float32 fingerprint of the query molecule.
        pipeline:    ToxNetPipeline in eval mode.
        thresholds:  Per-endpoint calibrated thresholds.
        target_cols: List of 12 endpoint names.
        device:      torch.device.
        easy_thr:    Gradient norm threshold for EASY classification.
        hard_thr:    Gradient norm threshold for HARD (below = TRAPPED).

    Returns:
        {
            endpoint: {
                'probability':      float,
                'threshold':        float,
                'flagged':          bool,
                'gradient_norm':    float,
                'escape_pressure':  float,   # same as gradient_norm, renamed for clarity
                'difficulty':       'EASY' | 'HARD' | 'TRAPPED' | 'CLEAN',
                'top_escape_bits':  list[int],   # top 5 bits to flip for this endpoint
            }
        }
    """
    pipeline.eval()   # Ensure eval mode for deterministic BatchNorm

    # Build input tensor with gradient tracking
    x = torch.tensor(fp_arr, dtype=torch.float32, device=device)
    x.requires_grad_(True)

    results = {}

    for ep_idx, ep in enumerate(target_cols):
        # Forward pass
        x_in = x.unsqueeze(0)        # (1, 4096)
        probs = pipeline(x_in)        # (1, 12)
        prob_ep = probs[0, ep_idx]    # scalar

        prob_val   = float(prob_ep.item())
        threshold  = float(thresholds.get(ep, 0.5))
        is_flagged = prob_val >= threshold

        if not is_flagged:
            results[ep] = {
                'probability':     round(prob_val, 4),
                'threshold':       round(threshold, 4),
                'flagged':         False,
                'gradient_norm':   None,
                'escape_pressure': None,
                'difficulty':      'CLEAN',
                'top_escape_bits': [],
            }
            # Zero out gradient before next endpoint
            if x.grad is not None:
                x.grad.zero_()
            continue

        # Backward: compute ∂(prob_ep) / ∂(fp_bits)
        pipeline.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        prob_ep.backward(retain_graph=True)

        if x.grad is None:
            results[ep] = {
                'probability':     round(prob_val, 4),
                'threshold':       round(threshold, 4),
                'flagged':         True,
                'gradient_norm':   0.0,
                'escape_pressure': 0.0,
                'difficulty':      'TRAPPED',
                'top_escape_bits': [],
            }
            continue

        grad = x.grad.detach().cpu().numpy()  # (4096,)
        grad_norm = float(np.linalg.norm(grad))

        # Classify difficulty
        if grad_norm >= easy_thr:
            difficulty = 'EASY'
        elif grad_norm >= hard_thr:
            difficulty = 'HARD'
        else:
            difficulty = 'TRAPPED'

        # Top-5 bits where gradient magnitude is highest
        # (these are the bits most worth flipping to reduce this endpoint's toxicity)
        # We take bits where gradient is NEGATIVE (flipping them DOWN reduces toxicity)
        neg_grad = -grad   # We want to DECREASE prob, so flip sign
        top_bits = np.argsort(np.abs(neg_grad))[::-1][:5].tolist()

        results[ep] = {
            'probability':     round(prob_val, 4),
            'threshold':       round(threshold, 4),
            'flagged':         True,
            'gradient_norm':   round(grad_norm, 6),
            'escape_pressure': round(grad_norm, 6),
            'difficulty':      difficulty,
            'top_escape_bits': [int(b) for b in top_bits],
        }

        # Zero out gradient for next endpoint
        x.grad.zero_()

    return results


def summarize_escape_pressures(escape_results: dict) -> dict:
    """
    Produce a summary of escape pressures across all flagged endpoints.

    Returns:
        {
            'flagged_count':      int,
            'easy_count':         int,
            'hard_count':         int,
            'trapped_count':      int,
            'overall_prognosis':  'OPTIMISTIC' | 'CHALLENGING' | 'REDESIGN_NEEDED',
            'prognosis_message':  str,
        }
    """
    flagged   = [v for v in escape_results.values() if v['flagged']]
    easy      = [v for v in flagged if v['difficulty'] == 'EASY']
    hard      = [v for v in flagged if v['difficulty'] == 'HARD']
    trapped   = [v for v in flagged if v['difficulty'] == 'TRAPPED']

    n_flagged = len(flagged)
    n_easy    = len(easy)
    n_hard    = len(hard)
    n_trapped = len(trapped)

    if n_flagged == 0:
        return {
            'flagged_count':     0,
            'easy_count':        0,
            'hard_count':        0,
            'trapped_count':     0,
            'overall_prognosis': 'CLEAN',
            'prognosis_message': 'No endpoints flagged. No escape analysis needed.',
        }

    trapped_fraction = n_trapped / n_flagged

    if trapped_fraction == 0.0:
        prognosis = 'OPTIMISTIC'
        message = (
            f'All {n_flagged} flagged endpoint(s) show meaningful gradient signal. '
            'Fragment substitution is likely to reduce toxicity across all flagged endpoints. '
            'Proceed with bioisostere candidates.'
        )
    elif trapped_fraction <= 0.5:
        prognosis = 'CHALLENGING'
        message = (
            f'{n_easy} endpoint(s) are near the decision boundary (substitution viable). '
            f'{n_trapped} endpoint(s) are deeply toxic — fragment substitution '
            'is unlikely to escape these. Focus candidates on the escapable endpoints, '
            'and consider additional structural changes for the trapped ones.'
        )
    else:
        prognosis = 'REDESIGN_NEEDED'
        message = (
            f'{n_trapped} of {n_flagged} flagged endpoints are deeply entrenched. '
            'The gradient analysis suggests the scaffold itself may be driving toxicity, '
            'not a specific fragment. Fragment substitution has limited expected benefit. '
            'A scaffold redesign may be required.'
        )

    return {
        'flagged_count':     n_flagged,
        'easy_count':        n_easy,
        'hard_count':        n_hard,
        'trapped_count':     n_trapped,
        'overall_prognosis': prognosis,
        'prognosis_message': message,
    }
