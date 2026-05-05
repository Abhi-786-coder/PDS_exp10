"""
Test script to run the full Phase 6 Prescription Pipeline end-to-end.
"""
import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import get_device
from src.explain import load_pipeline_from_artifact
from src.bioisostere import load_chembl_cache
from src.prescription_pipeline import run_prescription_pipeline

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

def main():
    print("Loading model and cache...")
    device = get_device()
    pipeline, artifact = load_pipeline_from_artifact('/workspace/models/model_artifact.pkl', device)
    thresholds = artifact.get('thresholds', {})
    
    X_fp_train = np.load('/workspace/data/processed/splits/X_fp_train.npy')
    # Use a small background sample from training set for SHAP
    X_train_bg = X_fp_train[:200]
    
    chembl_cache = load_chembl_cache()
    if not chembl_cache:
        print("ERROR: ChEMBL cache is empty. Run build_chembl_cache.py first.")
        return
    
    # We use a known toxic molecule. Aniline is often flagged for toxicity.
    test_smiles = 'Nc1ccccc1' # Aniline
    
    print(f"\nRunning prescription pipeline on: {test_smiles}")
    print("This will do prediction, SHAP, fragment mapping, ChEMBL lookup, and Pareto ranking...")
    
    result = run_prescription_pipeline(
        smiles=test_smiles,
        pipeline=pipeline,
        thresholds=thresholds,
        target_cols=TARGET_COLS,
        X_train_bg=X_train_bg,
        X_fp_train=X_fp_train,
        device=device,
        chembl_cache=chembl_cache,
        fp_radius=artifact.get('fp_radius', 2),
        fp_n_bits=artifact.get('fp_n_bits', 4096),
        top_shap_bits=3,
        max_candidates=10,
        shap_bg_size=100
    )
    
    print("\n--- PIPELINE RESULT ---")
    print(f"Status: {result['pipeline_status']}")
    if result['pipeline_status'] == 'complete':
        print(f"Flagged Endpoints: {result['prediction']['n_flagged']} / 12")
        print(f"Structural Alerts: PAINS={result['alerts']['pains_hit']}, Brenk={result['alerts']['brenk_hit']}")
        print(f"SHAP Confidence:   {result['alerts']['shap_confidence']}")
        print(f"Bioisosteres Found: {result['n_candidates_found']} raw -> {result['n_after_filter']} passing filters")
        
        print("\nPareto Candidates:")
        for c in result['pareto_candidates']:
            print(f"  Rank {c['rank']}: {c['smiles']}")
            print(f"    Mean Tox Prob: {c['mean_tox_prob']:.3f} | Flagged: {c['n_flagged']}")
            print(f"    Synth: {c['synth_verdict']} (SA={c['sa_score']}, SC={c['sc_score']})")
            print(f"    ADME: Passes={c['adme']['passes']} (ΔLogP={c['adme']['delta_logp']}, ΔMW={c['adme']['delta_mw']})")
            print(f"    OOD Warn: {c['ood_warning']} (max sim={c['ood_max_sim']})")
            print(f"    Pareto Score: {c['pareto_score']}")
            print("  ---")
    else:
        print(f"Result dump: {result}")

if __name__ == '__main__':
    main()
