"""
scripts/precompute_frameworks.py
One-time offline pre-computation for the three groundbreaking frameworks.

Run this ONCE before starting the API. It produces two artifacts:
  - data/stbi_artifact.pkl         (Scaffold Toxicity Brittleness Index)
  - data/constellation_artifact.pkl (Toxicity Constellations)

These files are loaded at API startup and used for every /analyze request.

Usage:
    cd Project_Implementation
    python scripts/precompute_frameworks.py

Expected runtime: 3–8 minutes on CPU (dominated by STBI Tanimoto computation).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.stbi          import compute_stbi_all_endpoints
from src.constellations import build_constellation_model
from src.train         import get_device
from src.explain       import load_pipeline_from_artifact

# ── Config ───────────────────────────────────────────────────────────────────

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

CLEANED_DATA_PATH     = os.path.join(PROJECT_ROOT, 'data', 'processed', 'tox21_cleaned.csv')
TRAIN_FP_PATH         = os.path.join(PROJECT_ROOT, 'data', 'processed', 'splits', 'X_fp_train.npy')
MODEL_ARTIFACT_PATH   = os.path.join(PROJECT_ROOT, 'models', 'model_artifact.pkl')

STBI_OUT_PATH         = os.path.join(PROJECT_ROOT, 'data', 'stbi_artifact.pkl')
CONST_OUT_PATH        = os.path.join(PROJECT_ROOT, 'data', 'constellation_artifact.pkl')

BATCH_SIZE            = 512   # For model inference on training set
N_CONSTELLATIONS      = 5


def run_model_on_training_set(pipeline, X_fp_train, device, batch_size=BATCH_SIZE):
    """
    Run ToxNetPipeline on the full training fingerprint matrix.
    Returns (N_train, 12) float32 probability matrix.
    """
    pipeline.eval()
    all_probs = []
    n = len(X_fp_train)
    print(f"  Running inference on {n} training molecules...")

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = torch.from_numpy(X_fp_train[start:end].astype(np.float32)).to(device)
            probs = pipeline(batch).cpu().numpy()   # (batch, 12)
            all_probs.append(probs)
            if (start // batch_size) % 5 == 0:
                print(f"    {end}/{n} done...")

    prob_matrix = np.vstack(all_probs).astype(np.float32)
    print(f"  Inference complete. Probability matrix shape: {prob_matrix.shape}")
    return prob_matrix


def main():
    print("=" * 60)
    print(" ToxNet Framework Pre-computation")
    print("=" * 60)

    # ── Validate paths ───────────────────────────────────────────────────────
    for path, name in [
        (CLEANED_DATA_PATH,   'tox21_cleaned.csv'),
        (TRAIN_FP_PATH,       'X_fp_train.npy'),
        (MODEL_ARTIFACT_PATH, 'model_artifact.pkl'),
    ]:
        if not os.path.exists(path):
            print(f"\n[ERROR] Required file not found: {path}")
            print(f"  Run the training notebooks first to generate {name}.")
            sys.exit(1)

    os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════
    # FRAMEWORK 1: STBI
    # ═══════════════════════════════════════════════════════════════════
    print("\n[1/3] Computing Scaffold Toxicity Brittleness Index (STBI)...")
    print("      This takes 2-5 minutes — computing Tanimoto distances per scaffold.")

    df = pd.read_csv(CLEANED_DATA_PATH)

    # Verify required columns exist
    missing_cols = [c for c in TARGET_COLS if c not in df.columns]
    if missing_cols:
        print(f"  [WARNING] Missing endpoint columns in CSV: {missing_cols}")
        TARGET_COLS_available = [c for c in TARGET_COLS if c in df.columns]
    else:
        TARGET_COLS_available = TARGET_COLS

    # Detect SMILES column name (may be 'smiles' or 'clean_smiles')
    smiles_col = None
    for candidate in ['smiles', 'clean_smiles', 'SMILES']:
        if candidate in df.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        print(f"[ERROR] No SMILES column found. Columns: {df.columns.tolist()}")
        sys.exit(1)

    # Normalise to 'smiles' so stbi.py can use a fixed column name
    if smiles_col != 'smiles':
        df = df.rename(columns={smiles_col: 'smiles'})
        print(f"  Renamed '{smiles_col}' → 'smiles' for STBI computation.")

    stbi_artifact = compute_stbi_all_endpoints(df, TARGET_COLS_available, fp_radius=2)

    with open(STBI_OUT_PATH, 'wb') as f:
        pickle.dump(stbi_artifact, f)
    print(f"  ✅ STBI artifact saved → {STBI_OUT_PATH}")
    print(f"     ({len(stbi_artifact)} scaffolds indexed)")

    # ═══════════════════════════════════════════════════════════════════
    # FRAMEWORK 2: Toxicity Constellations
    # ═══════════════════════════════════════════════════════════════════
    print("\n[2/3] Building Toxicity Constellations...")

    # Load model
    device = get_device()
    print(f"  Loading ToxNetPipeline on {device}...")
    pipeline, artifact = load_pipeline_from_artifact(MODEL_ARTIFACT_PATH, device)

    # Load training fingerprints
    print("  Loading training fingerprints...")
    X_fp_train = np.load(TRAIN_FP_PATH)
    print(f"  X_fp_train shape: {X_fp_train.shape}")

    # Run model inference on full training set → probability matrix
    print("  Running model inference to build probability matrix...")
    prob_matrix = run_model_on_training_set(pipeline, X_fp_train, device)

    # Save the prob matrix for reference (optional, useful for debugging)
    prob_matrix_path = os.path.join(PROJECT_ROOT, 'data', 'train_prob_matrix.npy')
    np.save(prob_matrix_path, prob_matrix)
    print(f"  Training probability matrix saved → {prob_matrix_path}")

    # Fit constellation model
    constellation_artifact = build_constellation_model(
        prob_matrix=prob_matrix,
        target_cols=TARGET_COLS,
        n_constellations=N_CONSTELLATIONS,
    )

    with open(CONST_OUT_PATH, 'wb') as f:
        pickle.dump(constellation_artifact, f)
    print(f"  ✅ Constellation artifact saved → {CONST_OUT_PATH}")

    # ═══════════════════════════════════════════════════════════════════
    # FRAMEWORK 3: MTEP Calibration Thresholds
    # ═══════════════════════════════════════════════════════════════════
    print("\n[3/3] Calibrating MTEP escape pressure thresholds...")
    print("      Computing gradient norms on a sample of toxic training molecules.")

    # Sample at most 200 toxic training molecules for calibration
    toxic_mask = (prob_matrix > 0.5).any(axis=1)
    toxic_idx  = np.where(toxic_mask)[0]
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(toxic_idx, size=min(200, len(toxic_idx)), replace=False)

    pipeline.eval()
    gradient_norms = []

    for idx in sample_idx:
        fp = X_fp_train[idx].astype(np.float32)
        x = torch.tensor(fp, dtype=torch.float32, device=device)
        x.requires_grad_(True)

        x_in = x.unsqueeze(0)
        probs = pipeline(x_in)

        # Use the endpoint with the highest predicted probability
        max_ep_idx = int(prob_matrix[idx].argmax())
        prob_ep = probs[0, max_ep_idx]

        pipeline.zero_grad()
        prob_ep.backward()

        if x.grad is not None:
            grad_norm = float(x.grad.detach().cpu().numpy().__abs__().sum()**0.5)
            gradient_norms.append(grad_norm)
            x.grad.zero_()

    if gradient_norms:
        gradient_norms = np.array(gradient_norms)
        # Thresholds: 66th and 33rd percentile of gradient norms on toxic training set
        easy_thr = float(np.percentile(gradient_norms, 66))
        hard_thr = float(np.percentile(gradient_norms, 33))

        print(f"  Gradient norm stats: min={gradient_norms.min():.4f}, "
              f"mean={gradient_norms.mean():.4f}, max={gradient_norms.max():.4f}")
        print(f"  EASY threshold (66th pct): {easy_thr:.4f}")
        print(f"  HARD threshold (33rd pct): {hard_thr:.4f}")

        # Save thresholds as part of constellation artifact (reuse same file)
        constellation_artifact['mtep_easy_thr'] = easy_thr
        constellation_artifact['mtep_hard_thr'] = hard_thr
        with open(CONST_OUT_PATH, 'wb') as f:
            pickle.dump(constellation_artifact, f)
        print(f"  ✅ MTEP thresholds saved into constellation artifact.")
    else:
        print("  [WARNING] Could not compute gradient norms. Using default thresholds.")

    # ═══════════════════════════════════════════════════════════════════
    # Done
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(" Pre-computation complete!")
    print("=" * 60)
    print(f"\n  STBI artifact:           {STBI_OUT_PATH}")
    print(f"  Constellation artifact:  {CONST_OUT_PATH}")
    print(f"  Training prob matrix:    {prob_matrix_path}")
    print("\n  You can now restart the API: python src/api.py")
    print("  The three new frameworks will be active in every /analyze response.")


if __name__ == '__main__':
    main()
