"""
src/featurize.py
Phase 2: Multi-Representation Featurization for Tox21.

Implements:
- ECFP4/ECFP6 Morgan fingerprints with bitInfo capture (critical for SHAP in Phase 6)
- RDKit 2D descriptor extraction
- StandardScaler fitting (train-only) to prevent data leakage

All 6 fingerprint configs are defined here per the roadmap ablation spec:
  radius ∈ {2, 3}  ×  n_bits ∈ {1024, 2048, 4096} = 6 combinations
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import StandardScaler
import warnings

# ─── Fingerprint Configurations (Ablation Study) ─────────────────────────────
# Best config feeds directly into ToxNet (Phase 4)
# Second-best config provides bitInfo dictionary for SHAP (Phase 6)
FP_CONFIGS = [
    {'radius': 2, 'n_bits': 1024, 'name': 'ECFP4_1024'},
    {'radius': 2, 'n_bits': 2048, 'name': 'ECFP4_2048'},  # Rogers & Hahn 2010 default
    {'radius': 2, 'n_bits': 4096, 'name': 'ECFP4_4096'},
    {'radius': 3, 'n_bits': 1024, 'name': 'ECFP6_1024'},
    {'radius': 3, 'n_bits': 2048, 'name': 'ECFP6_2048'},
    {'radius': 3, 'n_bits': 4096, 'name': 'ECFP6_4096'},  # Largest coverage
]


def smiles_to_morgan(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Returns a Morgan fingerprint bit vector as a numpy array.
    Returns a zero vector if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.uint8)


def smiles_to_morgan_with_info(smiles: str, radius: int = 2, n_bits: int = 2048) -> tuple:
    """
    Returns (fingerprint_array, bit_info_dict).

    ROADMAP WARNING: bit_info dict MUST be declared OUTSIDE the function call.
    If declared inside GetMorganFingerprintAsBitVect(), it gets garbage collected
    immediately, causing SHAP-to-atom mapping in Phase 6 to fail silently.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8), {}

    bit_info = {}  # ✔ Declared OUTSIDE the RDKit call
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=bit_info
    )
    return np.array(fp, dtype=np.uint8), bit_info


def smiles_to_rdkit_desc(smiles: str) -> np.ndarray:
    """
    Returns a vector of 200 RDKit 2D molecular descriptors.
    Returns an all-NaN vector if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(len(Descriptors.descList), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        desc_values = [func(mol) for _, func in Descriptors.descList]
    return np.array(desc_values, dtype=np.float32)


def build_fingerprint_matrix(smiles_list: list, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Build full fingerprint matrix for a list of SMILES strings."""
    return np.vstack([smiles_to_morgan(s, radius=radius, n_bits=n_bits) for s in smiles_list])


def build_descriptor_matrix(smiles_list: list) -> np.ndarray:
    """Build full descriptor matrix for a list of SMILES strings."""
    return np.vstack([smiles_to_rdkit_desc(s) for s in smiles_list])


class DescriptorScaler:
    """
    Wraps StandardScaler to enforce train-only fitting.

    ROADMAP CAUTION: Binary fingerprints [0,1] concatenated with raw continuous
    descriptors (e.g., MolWt in [18, 800]) will destroy PCA variance and PyTorch
    gradients. Always scale descriptors AFTER splitting, fitted ONLY on train set.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, X_train_desc: np.ndarray) -> np.ndarray:
        """Fit on train descriptors ONLY, then transform. Replaces NaN with 0 first."""
        X = np.nan_to_num(X_train_desc, nan=0.0)
        result = self.scaler.fit_transform(X)
        self._fitted = True
        return result

    def transform(self, X_desc: np.ndarray) -> np.ndarray:
        """Transform val/test/calib descriptors using train-fitted scaler."""
        if not self._fitted:
            raise RuntimeError("DescriptorScaler must be fitted on training data first.")
        X = np.nan_to_num(X_desc, nan=0.0)
        return self.scaler.transform(X)
