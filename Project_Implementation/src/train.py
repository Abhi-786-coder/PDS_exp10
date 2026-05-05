"""
src/train.py
Phase 4: Two-Pass Bootstrap Training + OPTUNA Hyperparameter Search.

Pipeline overview:
  Pass 1  — Train ToxNetLite on raw 4096-bit fingerprints (20 epochs, Focal Loss, no SMOTE)
           → Extract 256-dim continuous embeddings from the trained backbone
           → Apply ADASYN augmentation in embedding space (only for SMOTE-valid endpoints)
  Pass 2  — Train final ToxNet on augmented 256-dim embeddings (full epochs, Focal Loss)

OPTUNA — Hyperparameter search over 6 neural parameters on the validation set.
          Search runs on each (Pass 1 + Pass 2) pipeline to find globally optimal params.
          Early stopping if no AUPRC improvement for 10 consecutive trials.
          Maximum ceiling: 200 trials.
"""

import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
import warnings
import logging

from src.model import ToxNetLite, ToxNet
from src.dataset import make_dataloader
from src.focal_loss import PerEndpointFocalLoss

# Suppress Optuna's per-trial logging — we print our own summaries
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('optuna').setLevel(logging.WARNING)


# ─── Device Setup ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Returns CUDA device if available and functional, else CPU.
    Blackwell (sm_120) requires PyTorch 2.7+ and CUDA 12.8 in the container.
    """
    if torch.cuda.is_available():
        try:
            # Quick functional test — catches sm_120 kernel issues with old PyTorch
            test = torch.ones(1, device='cuda')
            _ = test + test
            device = torch.device('cuda')
            print(f"GPU active: {torch.cuda.get_device_name(0)}")
            return device
        except RuntimeError:
            print("WARNING: CUDA detected but not functional. Falling back to CPU.")
    print("Using CPU.")
    return torch.device('cpu')


# ─── Core Training Functions ───────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: PerEndpointFocalLoss,
    device: torch.device,
    clip_grad_norm: float = 1.0,
) -> float:
    """Single training epoch. Returns mean batch loss."""
    model.train()
    total_loss, n_batches = 0.0, 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # Faster than zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients with Focal Loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    target_cols: list,
    device: torch.device,
    batch_size: int = 256,
) -> tuple:
    """
    Evaluate model on validation set. Returns (macro_auprc, per_endpoint_dict).
    Uses batched inference for memory efficiency on large val sets.
    """
    model.eval()
    X_tensor = torch.from_numpy(X_val.astype(np.float32))

    # Batched inference
    all_probs = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device, non_blocking=True)
        probs = torch.sigmoid(model(batch)).cpu().numpy()
        all_probs.append(probs)
    Y_proba = np.vstack(all_probs)  # (n_val, 12)

    auprc_scores, per_endpoint = [], {}
    for i, col in enumerate(target_cols):
        y_true = Y_val[:, i]
        mask = ~np.isnan(y_true)
        if mask.sum() < 5 or len(np.unique(y_true[mask])) < 2:
            continue
        auprc = average_precision_score(y_true[mask], Y_proba[mask, i])
        auprc_scores.append(auprc)
        per_endpoint[col] = round(auprc, 4)

    macro_auprc = float(np.mean(auprc_scores)) if auprc_scores else 0.0
    return macro_auprc, per_endpoint


# ─── Pass 1: ToxNetLite Training ──────────────────────────────────────────────

def train_toxnet_lite(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    target_cols: list,
    params: dict,
    device: torch.device,
    n_epochs: int = 20,
    verbose: bool = True,
) -> tuple:
    """
    Train ToxNetLite on raw fingerprints for `n_epochs`.
    Returns (trained_model, best_val_auprc).

    This is Pass 1 of the Two-Pass Bootstrap.
    """
    pos_weights = PerEndpointFocalLoss.compute_pos_weights(Y_train, target_cols)

    model = ToxNetLite(
        input_dim=X_train.shape[1],
        shared_dims=[512, 256],
        head_dim=32,
        n_tasks=len(target_cols),
        dropout=params.get('dropout', 0.3),
    ).to(device)

    criterion = PerEndpointFocalLoss(
        pos_weights=pos_weights,
        target_names=target_cols,
        gamma=params.get('focal_gamma', 2.0),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get('learning_rate', 1e-3),
        weight_decay=params.get('weight_decay', 1e-4),
    )

    # Cosine annealing LR schedule — smoother convergence than fixed LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    loader = make_dataloader(
        X_train, Y_train,
        batch_size=params.get('batch_size', 64),
        shuffle=True,
    )

    best_auprc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            val_auprc, _ = evaluate(model, X_val, Y_val, target_cols, device)
            if verbose:
                print(f"  [Lite Ep {epoch+1:2d}/{n_epochs}] Loss={loss:.4f} | Val AUPRC={val_auprc:.4f}")
            if val_auprc > best_auprc:
                best_auprc = val_auprc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_auprc


# ─── Embedding Extraction + SMOTE Augmentation ─────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: ToxNetLite,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract 256-dim backbone embeddings from ToxNetLite."""
    model.eval()
    X_tensor = torch.from_numpy(X.astype(np.float32))
    embeddings = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device, non_blocking=True)
        emb = model.get_embeddings(batch).cpu().numpy()
        embeddings.append(emb)
    return np.vstack(embeddings)


def augment_embeddings(
    embeddings: np.ndarray,
    Y_train: np.ndarray,
    target_cols: list,
    smote_valid_map: dict,
) -> tuple:
    """
    Apply ADASYN augmentation in the continuous 256-dim embedding space
    for SMOTE-valid endpoints only. Returns (augmented_embeddings, augmented_Y).

    Why ADASYN over vanilla SMOTE:
      ADASYN adaptively generates more synthetic samples near the decision boundary
      (hard-to-classify toxic compounds) rather than uniformly. This is more
      appropriate for toxicity prediction where boundary cases matter most.

    Why followed by TomekLinks:
      Removes noisy majority-class samples that overlap with the augmented minority.
      Cleans the boundary rather than just adding minority samples.
    """
    n_orig = len(embeddings)
    n_endpoints = Y_train.shape[1]

    # Build a combined augmented dataset: start with originals
    X_aug = embeddings.copy()
    Y_aug = Y_train.copy()

    for i, col in enumerate(target_cols):
        if not smote_valid_map.get(col, False):
            continue  # Skip SMOTE-invalid endpoints (e.g., NR-AhR)

        y_ep = Y_train[:, i]
        valid_mask = ~np.isnan(y_ep)
        n_valid = valid_mask.sum()

        if n_valid < 20:
            continue  # Not enough data

        n_minority = int((y_ep[valid_mask] == 1).sum())
        if n_minority < 6:
            continue  # ADASYN needs at least 6 minority samples

        X_ep = embeddings[valid_mask]
        y_ep_clean = y_ep[valid_mask]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                k = min(5, n_minority - 1)
                adasyn = ADASYN(random_state=42, n_neighbors=k)
                X_res, y_res = adasyn.fit_resample(X_ep, y_ep_clean)

                tomek = TomekLinks()
                X_res, y_res = tomek.fit_resample(X_res, y_res)

            # Only keep the NEW synthetic samples (rows beyond the original count)
            n_new = len(X_res) - n_valid
            if n_new > 0:
                X_new = X_res[n_valid:]
                y_new = y_res[n_valid:]

                # Create new label rows for these synthetic molecules
                # Fill other endpoints with -1 (unknown/masked)
                Y_new_rows = np.full((n_new, n_endpoints), np.nan)
                Y_new_rows[:, i] = y_new

                X_aug = np.vstack([X_aug, X_new])
                Y_aug = np.vstack([Y_aug, Y_new_rows])

        except Exception as e:
            print(f"  ADASYN failed for {col}: {e}. Skipping.")
            continue

    n_added = len(X_aug) - n_orig
    print(f"  SMOTE augmentation: {n_orig} → {len(X_aug)} samples (+{n_added} synthetic)")
    return X_aug, Y_aug


# ─── Pass 2: Full ToxNet Training ─────────────────────────────────────────────

def train_toxnet_full(
    X_train_emb: np.ndarray,
    Y_train: np.ndarray,
    X_val_emb: np.ndarray,
    Y_val: np.ndarray,
    target_cols: list,
    params: dict,
    device: torch.device,
    n_epochs: int = 100,
    patience: int = 15,
    verbose: bool = True,
) -> tuple:
    """
    Train the final ToxNet on augmented 256-dim embeddings.
    Includes early stopping based on val AUPRC (patience=15 epochs).
    Returns (best_model, best_val_auprc).
    """
    pos_weights = PerEndpointFocalLoss.compute_pos_weights(Y_train, target_cols)

    model = ToxNet(
        input_dim=X_train_emb.shape[1],
        shared_dims=params.get('shared_dims', [256, 128]),
        head_dim=64,
        n_tasks=len(target_cols),
        dropout=params.get('dropout', 0.3),
    ).to(device)

    criterion = PerEndpointFocalLoss(
        pos_weights=pos_weights,
        target_names=target_cols,
        gamma=params.get('focal_gamma', 2.0),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get('learning_rate', 1e-3),
        weight_decay=params.get('weight_decay', 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    loader = make_dataloader(
        X_train_emb, Y_train,
        batch_size=params.get('batch_size', 64),
        shuffle=True,
        use_embedding=True,
    )

    best_auprc = 0.0
    best_state = None
    no_improve_count = 0

    for epoch in range(n_epochs):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()

        val_auprc, per_ep = evaluate(model, X_val_emb, Y_val, target_cols, device)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  [Full Ep {epoch+1:3d}/{n_epochs}] Loss={loss:.4f} | Val AUPRC={val_auprc:.4f}")

        if val_auprc > best_auprc + 1e-4:
            best_auprc = val_auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_auprc


# ─── OPTUNA Objective ─────────────────────────────────────────────────────────

def build_optuna_objective(
    X_train_fp: np.ndarray,
    Y_train: np.ndarray,
    X_val_fp: np.ndarray,
    Y_val: np.ndarray,
    target_cols: list,
    smote_valid_map: dict,
    device: torch.device,
):
    """
    Returns a closure (trial → macro_auprc) for Optuna to optimise.

    Search space (as per roadmap Phase 4):
      lr           : 1e-4 to 1e-2 (log scale)
      dropout      : 0.1 to 0.5
      batch_size   : 32 / 64 / 128
      focal_gamma  : 1.0 to 5.0
      shared_dims  : [256, 128] or [512, 256, 128]
      weight_decay : 1e-6 to 1e-3 (log scale)

    Each trial runs the FULL Two-Pass Bootstrap:
      Lite (20 epochs) → embeddings → SMOTE → Full (30 epochs fast search)
    Final training uses 100 epochs with the best found params.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            'learning_rate': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'dropout':       trial.suggest_float('dropout', 0.1, 0.5),
            'batch_size':    trial.suggest_categorical('batch_size', [32, 64, 128]),
            'focal_gamma':   trial.suggest_float('gamma', 1.0, 5.0),
            'shared_dims':   trial.suggest_categorical(
                                 'dims',
                                 ['256_128', '512_256_128']
                             ),
            'weight_decay':  trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }

        # Convert string representation back to list
        dims_map = {
            '256_128':     [256, 128],
            '512_256_128': [512, 256, 128],
        }
        params['shared_dims'] = dims_map[params['shared_dims']]

        try:
            # Pass 1: ToxNetLite on fingerprints
            lite_model, _ = train_toxnet_lite(
                X_train_fp, Y_train, X_val_fp, Y_val,
                target_cols, params, device,
                n_epochs=20, verbose=False,
            )

            # Extract embeddings + augment
            X_train_emb = extract_embeddings(lite_model, X_train_fp, device)
            X_val_emb   = extract_embeddings(lite_model, X_val_fp,   device)
            X_aug, Y_aug = augment_embeddings(X_train_emb, Y_train, target_cols, smote_valid_map)

            # Pass 2: Full ToxNet on augmented embeddings (30 epochs for speed during search)
            full_model, val_auprc = train_toxnet_full(
                X_aug, Y_aug, X_val_emb, Y_val,
                target_cols, params, device,
                n_epochs=30, patience=10, verbose=False,
            )

            return val_auprc

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            return 0.0

    return objective


# ─── Full Pipeline Orchestrator ────────────────────────────────────────────────

def run_full_pipeline(
    X_train_fp: np.ndarray,
    Y_train: np.ndarray,
    X_val_fp: np.ndarray,
    Y_val: np.ndarray,
    target_cols: list,
    smote_valid_map: dict,
    device: torch.device,
    n_optuna_trials: int = 200,
    n_final_epochs: int = 100,
) -> tuple:
    """
    Orchestrates the complete Phase 4 pipeline:
      1. OPTUNA search (n_optuna_trials, early stopping at 10 stagnant trials)
      2. Full Two-Pass training with best params for n_final_epochs
      3. Returns (lite_model, full_model, best_params, study)

    Returns:
      lite_model  : Trained ToxNetLite (for embedding extraction in inference)
      full_model  : Trained ToxNet (the final predictor)
      best_params : Dict of best hyperparameters found
      study       : Optuna study object (for convergence analysis)
    """
    print("=" * 60)
    print("PHASE 4: Two-Pass Bootstrap + OPTUNA")
    print(f"Device: {device}")
    print(f"OPTUNA trials: {n_optuna_trials} (early stop after 10 stagnant)")
    print("=" * 60)

    # ── OPTUNA Search ────────────────────────────────────────────────────────
    print("\n[Step 1] Running OPTUNA hyperparameter search...")
    objective = build_optuna_objective(
        X_train_fp, Y_train, X_val_fp, Y_val,
        target_cols, smote_valid_map, device,
    )

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='toxnet_phase4',
    )

    # Custom callback for progress reporting + early stopping
    best_so_far = [0.0]
    stagnant = [0]

    def callback(study, trial):
        if trial.value and trial.value > best_so_far[0] + 1e-4:
            best_so_far[0] = trial.value
            stagnant[0] = 0
            print(f"  Trial {trial.number:3d}: AUPRC={trial.value:.4f} ★ NEW BEST")
        else:
            stagnant[0] += 1
            if trial.number % 10 == 0:
                val_str = f"{trial.value:.4f}" if trial.value is not None else "N/A"
                print(f"  Trial {trial.number:3d}: AUPRC={val_str} (stagnant: {stagnant[0]}/10)")

        if stagnant[0] >= 10:
            print(f"\n  Early stopping: no improvement for 10 trials.")
            study.stop()

    study.optimize(objective, n_trials=n_optuna_trials, callbacks=[callback])

    best_params = study.best_params.copy()
    dims_map = {'256_128': [256, 128], '512_256_128': [512, 256, 128]}
    best_params['shared_dims'] = dims_map[best_params.pop('dims')]
    best_params['learning_rate'] = best_params.pop('lr')
    best_params['focal_gamma'] = best_params.pop('gamma')

    print(f"\n✅ OPTUNA complete: {len(study.trials)} trials")
    print(f"   Best Macro AUPRC: {study.best_value:.4f}")
    print(f"   Best params: {best_params}")

    # ── Final Two-Pass Training with Best Params ──────────────────────────────
    print(f"\n[Step 2] Final Pass 1 — ToxNetLite ({20} epochs)...")
    lite_model, lite_auprc = train_toxnet_lite(
        X_train_fp, Y_train, X_val_fp, Y_val,
        target_cols, best_params, device,
        n_epochs=20, verbose=True,
    )
    print(f"  ToxNetLite val AUPRC: {lite_auprc:.4f}")

    print(f"\n[Step 3] Extracting embeddings + SMOTE augmentation...")
    X_train_emb = extract_embeddings(lite_model, X_train_fp, device)
    X_val_emb   = extract_embeddings(lite_model, X_val_fp,   device)
    X_aug, Y_aug = augment_embeddings(X_train_emb, Y_train, target_cols, smote_valid_map)

    print(f"\n[Step 4] Final Pass 2 — Full ToxNet ({n_final_epochs} epochs)...")
    full_model, final_auprc = train_toxnet_full(
        X_aug, Y_aug, X_val_emb, Y_val,
        target_cols, best_params, device,
        n_epochs=n_final_epochs, patience=15, verbose=True,
    )
    print(f"\n Final ToxNet val AUPRC: {final_auprc:.4f}")

    return lite_model, full_model, best_params, study, X_val_emb
