"""
Offline script to build the ChEMBL cache for the prescription pipeline.
Run this ONCE. It takes ~2 minutes depending on API limits.
"""
import sys
import os
import pickle

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bioisostere import build_chembl_cache

# A set of common, simple fragments that SHAP might flag,
# and some standard bioisostere replacements.
# Note: In a real production system, this would be thousands of fragments.
FRAGMENT_QUERIES = [
    'Nc1ccccc1',     # Aniline (often toxic, flagged by model)
    '[NX3](=O)=O',   # Nitro group (often toxic)
    'c1ccc(F)cc1',   # Fluorobenzene (common safe replacement)
    'c1ccc(O)cc1',   # Phenol
    'C1CCCCC1',      # Cyclohexane
    'c1ccncc1',      # Pyridine
    'c1cncnc1',      # Pyrimidine
    'C(F)(F)F',      # Trifluoromethyl
    'C#N',           # Nitrile
    'c1ccccc1'       # Benzene ring
]

if __name__ == '__main__':
    print("Building local ChEMBL cache...")
    cache_path = '/workspace/data/chembl_cache.pkl'
    
    # We cap at 30 per fragment for speed during this demonstration
    cache = build_chembl_cache(
        fragment_smiles_list=FRAGMENT_QUERIES,
        cache_path=cache_path,
        max_per_fragment=30
    )
    
    total_mols = sum(len(mols) for mols in cache.values())
    print(f"Done. Cache contains {total_mols} molecules across {len(cache)} fragments.")
