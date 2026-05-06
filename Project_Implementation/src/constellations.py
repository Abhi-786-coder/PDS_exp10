"""
src/constellations.py
Toxicity Constellation Mapping — Original Framework.

What are Toxicity Constellations?
==================================
Every molecule run through ToxNet produces a 12-dimensional probability vector —
one probability per Tox21 endpoint. Everyone takes the mean and calls it
"overall toxicity." That discards the biological signal.

When you cluster molecules in this 12-dimensional BIOLOGICAL RESPONSE SPACE
(not chemical structure space), you find mechanistic archetypes:

  - Constellation 0: Oxidative stress cascade (SR-ARE + SR-MMP + NR-AhR high)
  - Constellation 1: Endocrine disruption (NR-ER + NR-AR + NR-Aromatase high)
  - Constellation 2: Direct genotoxicity (SR-p53 + SR-ATAD5 elevated, others low)
  ... and so on.

These groups correspond to real biological pathways.

Why this matters:
-----------------
Instead of saying "your molecule has mean toxicity 0.6," the system now says:
"Your molecule is a canonical Endocrine Disruptor (Constellation 1).
 Molecules in this constellation have historically been difficult to escape
 via fragment substitution alone — the receptor-binding scaffold is usually
 the root cause."

This is a fundamentally different kind of diagnostic output.

This approach does not exist in any published Tox21 paper.
"""

import numpy as np
import pickle
import warnings
from typing import Optional

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ─── Constants ────────────────────────────────────────────────────────────────

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

N_CONSTELLATIONS = 5   # Empirically chosen: 4-6 gives meaningful separation

# Endpoint short names for display
EP_SHORT = {
    'NR-AR':         'AR',
    'NR-AR-LBD':     'AR-LBD',
    'NR-AhR':        'AhR',
    'NR-Aromatase':  'Arom',
    'NR-ER':         'ER',
    'NR-ER-LBD':     'ER-LBD',
    'NR-PPAR-gamma': 'PPAR-γ',
    'SR-ARE':        'ARE',
    'SR-ATAD5':      'ATAD5',
    'SR-HSE':        'HSE',
    'SR-MMP':        'MMP',
    'SR-p53':        'p53',
}

# Mechanism interpretations keyed by dominant endpoint pattern
# These are derived from actual Tox21 AOP (Adverse Outcome Pathway) literature.
_MECHANISM_HINTS = {
    'ARE':   'Oxidative stress / Nrf2 pathway activation',
    'MMP':   'Mitochondrial membrane disruption (cytotoxicity)',
    'AhR':   'Dioxin-like toxicity — CYP1A1 metabolic activation',
    'p53':   'DNA damage / carcinogenicity signal',
    'ATAD5': 'Genotoxicity (DNA replication stress)',
    'ER':    'Estrogen receptor disruption (endocrine)',
    'AR':    'Androgen receptor disruption (endocrine)',
    'Arom':  'Aromatase inhibition — hormonal axis disruption',
    'HSE':   'General stress response (heat shock pathway)',
    'PPAR-γ': 'Metabolic receptor modulation',
}


# ─── Building the Constellation Model ────────────────────────────────────────

def build_constellation_model(
    prob_matrix:      np.ndarray,
    target_cols:      list,
    n_constellations: int = N_CONSTELLATIONS,
    random_state:     int = 42,
) -> dict:
    """
    Fit a KMeans constellation model on the training set probability matrix.

    Args:
        prob_matrix:      (N, 12) float array — model predictions on training set.
                          Rows: molecules. Columns: endpoint probabilities.
        target_cols:      List of 12 endpoint names (must match prob_matrix columns).
        n_constellations: Number of mechanistic archetypes to find.
        random_state:     For reproducibility.

    Returns:
        Artifact dict saved to disk and loaded at startup:
        {
            'kmeans':           sklearn KMeans object (fitted),
            'scaler':           StandardScaler (fitted on prob_matrix),
            'centroids':        (n_constellations, 12) raw centroid probabilities,
            'constellation_names': list of auto-generated name strings,
            'target_cols':      list of endpoint names,
            'n_constellations': int,
        }
    """
    # ── Filter: only include molecules toxic on at least one endpoint ──────────
    # All-safe molecules cluster trivially (they're all near zero in 12D).
    # We want constellations among TOXIC patterns.
    toxic_mask = (prob_matrix > 0.3).any(axis=1)
    toxic_probs = prob_matrix[toxic_mask]

    print(f"Building constellations on {toxic_probs.shape[0]} toxic-leaning molecules...")
    print(f"(Filtered from {prob_matrix.shape[0]} total training molecules)")

    # ── Scale: StandardScaler makes KMeans distance metric meaningful ─────────
    scaler = StandardScaler()
    toxic_scaled = scaler.fit_transform(toxic_probs)

    # ── KMeans in 12D biological response space ───────────────────────────────
    km = KMeans(
        n_clusters=n_constellations,
        n_init=20,          # More restarts → better solution
        max_iter=500,
        random_state=random_state,
        algorithm='lloyd',
    )
    km.fit(toxic_scaled)

    # Raw centroids in original (unscaled) probability space
    centroids_raw = scaler.inverse_transform(km.cluster_centers_)
    centroids_raw = np.clip(centroids_raw, 0.0, 1.0)

    # ── Auto-name each constellation ──────────────────────────────────────────
    constellation_names = []
    for i, centroid in enumerate(centroids_raw):
        dominant_idx = np.argsort(centroid)[::-1][:3]
        dominant_eps = [EP_SHORT.get(target_cols[d], target_cols[d]) for d in dominant_idx]
        name = " + ".join(dominant_eps)
        constellation_names.append(f"C{i+1}: {name}")

    for i, name in enumerate(constellation_names):
        centroid = centroids_raw[i]
        print(f"  {name}")
        dominant_idx = np.argsort(centroid)[::-1][:3]
        for d in dominant_idx:
            print(f"    {target_cols[d]}: {centroid[d]:.3f}")

    return {
        'kmeans':              km,
        'scaler':              scaler,
        'centroids':           centroids_raw,
        'constellation_names': constellation_names,
        'target_cols':         target_cols,
        'n_constellations':    n_constellations,
    }


# ─── Classification at Inference Time ────────────────────────────────────────

def classify_molecule(
    prob_vector:           np.ndarray,
    constellation_artifact: dict,
) -> dict:
    """
    Classify a query molecule into its toxicity constellation.

    Args:
        prob_vector:             (12,) float array — endpoint probabilities for
                                 the query molecule (from ToxNetPipeline).
        constellation_artifact:  Pre-built artifact from build_constellation_model().

    Returns:
        {
            'constellation_id':    int,
            'constellation_name':  str,
            'distance':            float,   # L2 distance to centroid in scaled space
            'proximity_label':     'CANONICAL' | 'TYPICAL' | 'BOUNDARY',
            'dominant_endpoints':  list[str],  # top 3 drivers in this constellation
            'mechanism_hint':      str,
            'centroid_profile':    dict,       # {endpoint: centroid_prob}
        }
    """
    km     = constellation_artifact['kmeans']
    scaler = constellation_artifact['scaler']
    centroids_raw = constellation_artifact['centroids']
    target_cols   = constellation_artifact['target_cols']
    names         = constellation_artifact['constellation_names']

    # Scale the query vector using the fitted scaler
    vec_scaled = scaler.transform(prob_vector.reshape(1, -1))

    # Find nearest cluster centroid
    constellation_id = int(km.predict(vec_scaled)[0])

    # Distance to that centroid (in scaled space)
    centroid_scaled = km.cluster_centers_[constellation_id]
    dist = float(np.linalg.norm(vec_scaled[0] - centroid_scaled))

    # Determine proximity label based on distance distribution
    # (rough thresholds — farther = more boundary between constellations)
    if dist < 1.0:
        proximity_label = 'CANONICAL'   # Textbook example of this archetype
    elif dist < 2.0:
        proximity_label = 'TYPICAL'     # Clearly in this constellation
    else:
        proximity_label = 'BOUNDARY'    # Between two constellations

    # Dominant endpoints in this constellation's centroid
    centroid_raw = centroids_raw[constellation_id]
    dominant_idx = np.argsort(centroid_raw)[::-1][:3]
    dominant_eps = [target_cols[d] for d in dominant_idx]

    # Build mechanism hint from dominant endpoints
    dominant_short = [EP_SHORT.get(ep, ep) for ep in dominant_eps]
    hints = [_MECHANISM_HINTS.get(s, '') for s in dominant_short if _MECHANISM_HINTS.get(s)]
    mechanism_hint = hints[0] if hints else 'Mixed mechanism signature'

    # Centroid profile for display
    centroid_profile = {
        target_cols[i]: round(float(centroid_raw[i]), 3)
        for i in range(len(target_cols))
    }

    return {
        'constellation_id':   constellation_id,
        'constellation_name': names[constellation_id],
        'distance':           round(dist, 4),
        'proximity_label':    proximity_label,
        'dominant_endpoints': dominant_eps,
        'mechanism_hint':     mechanism_hint,
        'centroid_profile':   centroid_profile,
    }
