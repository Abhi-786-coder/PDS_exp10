"""
src/explain.py
Phase 5: Threshold Calibration + SHAP Explainability.

Design decisions:

THRESHOLD CALIBRATION
  We use the CALIBRATION split (780 molecules), never the test set.
  Using the test set for threshold selection is data leakage — it would
  inflate test metrics and give a false sense of performance.
  For each of the 12 endpoints independently, we find the threshold that
  maximises F1 score. F1 is preferred over accuracy because the classes are
  heavily imbalanced (2–17% toxic). A model predicting all-negative would
  have >95% accuracy but 0.0 AUPRC.

SHAP
  Our Two-Pass architecture has two separate models:
    - ToxNetLite: 4096-bit FP → 256-dim embedding
    - ToxNet:     256-dim embedding → 12 logits
  SHAP needs a single end-to-end model. We create a ToxNetPipeline wrapper
  that chains both into one forward call: 4096 FP → prediction.

  We use shap.GradientExplainer (Expected Gradients) rather than DeepExplainer
  because:
    - Expected Gradients is more numerically stable with modern PyTorch.
    - It satisfies the same Shapley axioms as DeepExplainer.
    - It handles BatchNorm layers more reliably during backward passes.

  SHAP values are computed in the original 4096-bit fingerprint space, giving
  us a per-bit importance score for each of the 12 endpoints. We then map the
  top-k bits back to SMARTS substructures using RDKit's bitInfo dictionaries.
"""

import numpy as np
import torch
import torch.nn as nn
import shap
import pickle
import warnings
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.utils import resample

from src.model import ToxNetLite, ToxNet


# ─── Combined Pipeline Model ───────────────────────────────────────────────────

class ToxNetPipeline(nn.Module):
    """
    Single end-to-end wrapper: 4096-bit FP → 12 sigmoid probabilities.
    Chains ToxNetLite backbone → ToxNet forward in one call.
    Required for SHAP attribution in the original fingerprint space.
    """

    def __init__(self, lite_model: ToxNetLite, full_model: ToxNet):
        super().__init__()
        self.lite_backbone = lite_model.backbone   # 4096 → 256
        self.full_model = full_model               # 256 → 12

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 12) sigmoid probabilities."""
        embeddings = self.lite_backbone(x)
        logits = self.full_model(embeddings)
        return torch.sigmoid(logits)


def load_pipeline_from_artifact(artifact_path: str, device: torch.device) -> ToxNetPipeline:
    """
    Loads the complete ToxNetPipeline from the saved model_artifact.pkl bundle.
    This is the canonical way to load the model for inference and explanation.
    """
    with open(artifact_path, 'rb') as f:
        artifact = pickle.load(f)

    lite_cfg = artifact['lite_model_config']
    full_cfg = artifact['full_model_config']

    lite_model = ToxNetLite(
        input_dim=lite_cfg['input_dim'],
        shared_dims=lite_cfg['shared_dims'],
    )
    lite_model.load_state_dict(artifact['lite_model_state'])
    lite_model.eval()

    full_model = ToxNet(
        input_dim=full_cfg['input_dim'],
        shared_dims=full_cfg['shared_dims'],
    )
    full_model.load_state_dict(artifact['full_model_state'])
    full_model.eval()

    pipeline = ToxNetPipeline(lite_model, full_model).to(device)
    pipeline.eval()
    return pipeline, artifact


# ─── Threshold Calibration ──────────────────────────────────────────────────────

@torch.no_grad()
def calibrate_thresholds(
    pipeline: ToxNetPipeline,
    X_calib: np.ndarray,
    Y_calib: np.ndarray,
    target_cols: list,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """
    Find the optimal classification threshold for each endpoint that maximises
    F1 score on the calibration set.

    Why F1 and not AUPRC?
      AUPRC is a ranking metric (threshold-free). F1 is a decision metric
      (threshold-dependent). We optimise F1 here because the downstream
      prescription validator needs a binary decision: "Toxic" or "Safe".

    Returns:
      dict mapping endpoint_name → optimal_threshold (float, 0.0–1.0)
    """
    # Get predictions on calibration set
    X_tensor = torch.from_numpy(X_calib.astype(np.float32))
    all_probs = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device, non_blocking=True)
        probs = pipeline(batch).cpu().numpy()
        all_probs.append(probs)
    Y_proba = np.vstack(all_probs)   # (n_calib, 12)

    thresholds = {}
    print("\nThreshold Calibration (Calibration Split)")
    print("-" * 50)

    for i, col in enumerate(target_cols):
        y_true = Y_calib[:, i]
        mask = ~np.isnan(y_true)

        if mask.sum() < 10 or len(np.unique(y_true[mask])) < 2:
            thresholds[col] = 0.5
            print(f"  {col:<20}: threshold=0.500 (insufficient data, default)")
            continue

        y_true_clean = y_true[mask].astype(int)
        y_prob_clean = Y_proba[mask, i]

        # Sweep thresholds from 0.05 to 0.95
        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.05, 0.96, 0.01):
            y_pred = (y_prob_clean >= thr).astype(int)
            if y_pred.sum() == 0:
                continue
            f1 = f1_score(y_true_clean, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        thresholds[col] = round(float(best_thr), 3)
        minority_pct = y_true_clean.mean() * 100
        print(f"  {col:<20}: threshold={best_thr:.3f}  (F1={best_f1:.3f}, toxic={minority_pct:.1f}%)")

    return thresholds


# ─── SHAP Explainability ────────────────────────────────────────────────────────

def compute_shap_values(
    pipeline: ToxNetPipeline,
    X_train_bg: np.ndarray,
    X_explain: np.ndarray,
    device: torch.device,
    n_background: int = 150,
    n_explain: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute SHAP Expected Gradient values for each molecule × endpoint × bit.

    Args:
        pipeline     : ToxNetPipeline (end-to-end model).
        X_train_bg   : Training fingerprints (used as background distribution).
        X_explain    : Molecules to explain (calibration set recommended).
        device       : CUDA or CPU.
        n_background : Number of background molecules for Expected Gradients.
        n_explain    : Number of molecules to explain (more = slower).

    Returns:
        shap_values: (n_explain, 12, 4096) numpy array.
                     shap_values[i, j, k] = importance of bit k
                     for molecule i predicting endpoint j.
    """
    pipeline.eval()

    # Sample background molecules
    rng = np.random.default_rng(random_state)
    bg_idx = rng.choice(len(X_train_bg), size=min(n_background, len(X_train_bg)), replace=False)
    X_bg = X_train_bg[bg_idx].astype(np.float32)

    # Sample molecules to explain
    exp_idx = rng.choice(len(X_explain), size=min(n_explain, len(X_explain)), replace=False)
    X_exp = X_explain[exp_idx].astype(np.float32)

    # Move to device
    bg_tensor  = torch.from_numpy(X_bg).to(device)
    exp_tensor = torch.from_numpy(X_exp).to(device)

    print(f"  Background: {len(X_bg)} molecules | Explaining: {len(X_exp)} molecules")
    print("  Running SHAP GradientExplainer (this takes 2–5 minutes)...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.GradientExplainer(pipeline, bg_tensor)
        # Returns list of 12 arrays, each (n_explain, 4096)
        raw_shap  = explainer.shap_values(exp_tensor)

    # raw_shap is a list of 12 arrays each (n_explain, 4096)
    # Stack into (n_explain, 12, 4096)
    if isinstance(raw_shap, list):
        shap_arr = np.stack(raw_shap, axis=1)   # (n_explain, 12, 4096)
    else:
        shap_arr = raw_shap   # Fallback

    print(f"  SHAP complete. Output shape: {shap_arr.shape}")
    return shap_arr, exp_idx


def get_top_shap_bits(
    shap_values: np.ndarray,
    endpoint_idx: int,
    top_k: int = 20,
    agg: str = 'mean_abs',
) -> list:
    """
    Get the top-k most important fingerprint bits for a given endpoint.

    Args:
        shap_values   : (n_explain, 12, 4096) array.
        endpoint_idx  : Which of the 12 endpoints to analyse.
        top_k         : Number of top bits to return.
        agg           : Aggregation method. 'mean_abs' = mean absolute SHAP
                        across all explained molecules. Best for global importance.

    Returns:
        List of dicts: [{'bit': int, 'importance': float}, ...]
    """
    ep_shap = shap_values[:, :, endpoint_idx]   # (n_explain, 4096)

    if agg == 'mean_abs':
        importance = np.abs(ep_shap).mean(axis=0)
    elif agg == 'mean':
        importance = ep_shap.mean(axis=0)
    else:
        importance = np.abs(ep_shap).mean(axis=0)

    top_bits = np.argsort(importance)[::-1][:top_k]
    return [{'bit': int(b), 'importance': float(importance[b])} for b in top_bits]


def build_global_shap_summary(
    shap_values: np.ndarray,
    target_cols: list,
    top_k: int = 15,
) -> dict:
    """
    Build a global SHAP summary: top-k important bits per endpoint.

    Returns:
        Dict mapping endpoint_name → list of top-bit dicts.
    """
    summary = {}
    for i, col in enumerate(target_cols):
        summary[col] = get_top_shap_bits(shap_values, i, top_k=top_k)
    return summary
