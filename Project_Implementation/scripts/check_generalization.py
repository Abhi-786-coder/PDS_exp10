"""
Check which molecules are NOT in the Tox21 training set,
then test our model on them and compare against known scientific literature.
"""
import sys, json
sys.path.insert(0, '/workspace')
import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/data/raw/tox21.csv')
smiles_col = df.columns[-1]
all_smiles = set(df[smiles_col].dropna().tolist())

# Molecules with KNOWN published toxicity profiles for Tox21 endpoints
# Source: peer-reviewed literature
test_molecules = [
    # (name, smiles, expected_flags, scientific_evidence)
    ("Benzo[a]pyrene", "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
     ["NR-AhR"], "Classic AhR agonist, IARC Group 1 carcinogen"),
    ("Genistein (soy phytoestrogen)", "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
     ["NR-ER"], "Well-documented ER agonist (IC50 ~1uM), published in thousands of papers"),
    ("Bisphenol S (BPS, BPA replacement)", "O=S(=O)(c1ccc(O)cc1)c1ccc(O)cc1",
     ["NR-ER", "NR-AR"], "Androgenic+estrogenic, Molina-Molina et al. 2013"),
    ("Zearalenone (fungal mycotoxin)", "O=C1CCCCCC(=O)CCCC/C=C/c2cc(O)cc(O)c21",
     ["NR-ER"], "Potent mycoestrogen, EC50 ~10nM on ER, EFSA assessed"),
    ("Resveratrol (red wine)", "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
     ["NR-ER"], "Weak phytoestrogen, Ki ~7uM on ERalpha"),
    ("Paxlovid (COVID drug, invented 2020)", "CC1(C2C1C(N(C2)C(=O)C(C(C)(C)C)NC(=O)C(F)(F)F)C(=O)NC(CC3CCNC3=O)C#N)C",
     [], "FDA approved 2021, NOT in 2014 Tox21 dataset"),
    ("Ibuprofen (common painkiller)", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
     [], "NSAID, no significant endocrine disruption at therapeutic doses"),
]

print("=== Generalization Test: Molecules NOT in Tox21 Training Set ===\n")

import urllib.request

for name, smiles, expected, evidence in test_molecules:
    in_set = smiles in all_smiles
    status = "IN DATASET" if in_set else "NOT IN DATASET"

    # Call the API
    try:
        req = urllib.request.Request(
            'http://localhost:8000/analyze',
            data=json.dumps({'smiles': smiles}).encode(),
            headers={'Content-Type': 'application/json'}
        )
        res = urllib.request.urlopen(req, timeout=30)
        data = json.loads(res.read())
        flagged = [ep for ep, f in data['prediction']['flags'].items() if f]
        n_flagged = data['prediction']['n_flagged']
        mean_prob = data['prediction']['mean_prob']
    except Exception as e:
        flagged, n_flagged, mean_prob = [], -1, -1

    # Judge correctness
    if not expected:
        correct = n_flagged == 0
        verdict = "CORRECT (safe)" if correct else "WRONG (false positive)"
    else:
        hits = [ep for ep in expected if ep in flagged]
        correct = len(hits) > 0
        verdict = f"CORRECT (caught {hits})" if correct else f"WRONG (missed {expected})"

    print(f"Molecule : {name}")
    print(f"Status   : {status}")
    print(f"Expected : {expected if expected else 'Safe (0 flags)'}")
    print(f"Got      : {flagged} ({n_flagged} flags, mean_prob={mean_prob})")
    print(f"Verdict  : {verdict}")
    print(f"Evidence : {evidence}")
    print()
