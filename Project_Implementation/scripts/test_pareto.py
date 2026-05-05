"""Test the full Pareto pipeline end-to-end via the FastAPI /analyze endpoint."""

import json
import urllib.request

# BPA — known endocrine disruptor, should flag NR-ER
BPA_SMILES = "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"
# DES — banned synthetic estrogen
DES_SMILES  = "CC/C(=C(\\CC)/c1ccc(O)cc1)/c1ccc(O)cc1"

def test_molecule(smiles, label):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"SMILES:  {smiles}")
    print('='*60)

    req = urllib.request.Request(
        'http://localhost:8000/analyze',
        data=json.dumps({'smiles': smiles}).encode(),
        headers={'Content-Type': 'application/json'}
    )

    try:
        res = urllib.request.urlopen(req, timeout=120)
        data = json.loads(res.read())

        status      = data["pipeline_status"]
        n_flagged   = data["prediction"]["n_flagged"]
        mean_prob   = data["prediction"]["mean_prob"]
        shap_bits   = data["shap_top_bits"]
        n_found     = data["n_candidates_found"]
        n_filtered  = data["n_after_filter"]
        candidates  = data["pareto_candidates"]

        print(f"Pipeline Status  : {status}")
        print(f"Endpoints Flagged: {n_flagged} / 12")
        print(f"Mean Tox Prob    : {mean_prob}")
        print(f"SHAP Top Bits    : {shap_bits}")
        print(f"Candidates Found : {n_found}")
        print(f"After Filter     : {n_filtered}")
        print(f"Pareto Ranked    : {len(candidates)}")

        if candidates:
            print("\n--- Top 3 Pareto Candidates ---")
            for c in candidates[:3]:
                smiles_short = c['smiles'][:55] + "..." if len(c['smiles']) > 55 else c['smiles']
                print(f"  Rank #{c['rank']} | {c['chembl_id']}")
                print(f"    SMILES      : {smiles_short}")
                print(f"    Mean Tox    : {c['mean_tox_prob']} | N Flagged: {c['n_flagged']}")
                print(f"    SA/SC Score : {c['sa_score']} / {c['sc_score']}")
                print(f"    Synth       : {c['synth_verdict']}")
                print(f"    ADME Passes : {c['adme'].get('passes')} | dLogP={c['adme'].get('delta_logp')} dMW={c['adme'].get('delta_mw')}")
                print(f"    Pareto Score: {c['pareto_score']} | Front: {c['pareto_front']}")
                print(f"    OOD Warning : {c['ood_warning']} (max_sim={c['ood_max_sim']})")
                print()
        else:
            print("\n  [!] NO PARETO CANDIDATES RETURNED")
            print(f"  Alerts: {data['alerts']}")

    except Exception as e:
        print(f"ERROR calling API: {e}")


if __name__ == "__main__":
    test_molecule(BPA_SMILES, "BPA (Bisphenol A)")
    test_molecule(DES_SMILES, "DES (Diethylstilbestrol)")
