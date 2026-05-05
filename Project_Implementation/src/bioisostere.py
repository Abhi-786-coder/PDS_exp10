"""
src/bioisostere.py
Phase 6 — Step 2/3: Fragment mapping + ChEMBL bioisostere retrieval.

Design Decisions
================

SYNTHESIZABILITY CONSENSUS (2-metric, peer-reviewed)
  We use SAScore + SCScore. SYBA (3rd metric) was intended but has a Python 3.11
  gzip incompatibility with its bundled data file. Two metrics still constitute
  rigorous consensus scoring per the literature:

  | Metric  | Source                           | Pass condition  |
  |---------|----------------------------------|-----------------|
  | SAScore | Ertl & Schuffenhauer (2009) JCIM | < 3.0           |
  | SCScore | Coley et al. (2018) JCIM         | < 3.0           |

  Verdict (2-metric):
    Both pass  → HIGH_CONFIDENCE
    1 passes   → SYNTHESIZABLE
    0 pass     → REJECT

CHEMBL CACHE STRATEGY
  ChEMBL enforces rate limiting. Live queries inside a UI freeze the app.
  We build a local pickle cache ONCE (offline), then query it at runtime.
  The cache is keyed by the SMARTS/SMILES fragment string.

FRAGMENT MAPPING
  SHAP gives us a bit index in the 4096-bit ECFP space. We use RDKit's
  Morgan fingerprint bitInfo dictionary to map each bit back to the
  (atom_index, radius) pair that produced it, then extract the substructure
  at that radius as a SMARTS pattern for ChEMBL querying.

  The bitInfo dict is produced during featurization and must be stored
  alongside the fingerprint for this to work. We reconstruct it on-the-fly
  from the input SMILES here.
"""

import os
import sys
import pickle
import warnings
import numpy as np
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, RDConfig
from rdkit.Chem import rdMolDescriptors

# SAScore — built into RDKit contrib, no extra install
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # type: ignore

# SCScore — cloned from connorcoley/scscore (MIT license)
_SCSCORE_ROOT = os.path.join(os.path.dirname(__file__), '..', 'scscore')
sys.path.insert(0, os.path.abspath(_SCSCORE_ROOT))

_SCSCORER = None  # Lazy-load — 300ms startup cost, only pay it once.

SCSCORE_MODEL = os.path.join(
    _SCSCORE_ROOT, 'models',
    'full_reaxys_model_1024bool',
    'model.ckpt-10654.as_numpy.json.gz'
)


def _get_scscorer():
    """Lazy-load SCScorer singleton (avoids repeated 300ms loads)."""
    global _SCSCORER
    if _SCSCORER is None:
        from scscore.standalone_model_numpy import SCScorer  # type: ignore
        _SCSCORER = SCScorer()
        _SCSCORER.restore(SCSCORE_MODEL)
    return _SCSCORER


# ─── Synthesizability Consensus ─────────────────────────────────────────────────

def synthesizability_consensus(smiles: str) -> dict:
    """
    2-metric synthesizability consensus (SAScore + SCScore).

    Returns:
        {
            'verdict':   'HIGH_CONFIDENCE' | 'SYNTHESIZABLE' | 'REJECT',
            'include':   bool,
            'sa_score':  float,
            'sc_score':  float,
            'votes':     '2/2' | '1/2' | '0/2',
        }
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'verdict': 'REJECT', 'include': False, 'sa_score': None,
                'sc_score': None, 'votes': '0/2'}

    # SAScore: Ertl & Schuffenhauer (2009) — < 3.0 = "easy to synthesize"
    try:
        sa = sascorer.calculateScore(mol)
        sa_ok = sa < 3.0
    except Exception:
        sa, sa_ok = 99.0, False

    # SCScore: Coley et al. (2018) — < 3.0 = below-midpoint complexity
    try:
        scscorer = _get_scscorer()
        _, sc = scscorer.get_score_from_smi(smiles)
        sc_ok = sc < 3.0
    except Exception:
        sc, sc_ok = 99.0, False

    votes = int(sa_ok) + int(sc_ok)
    verdicts = {
        2: ('HIGH_CONFIDENCE', True),
        1: ('SYNTHESIZABLE', True),
        0: ('REJECT', False),
    }
    verdict, include = verdicts[votes]
    return {
        'verdict':  verdict,
        'include':  include,
        'sa_score': round(float(sa), 3),
        'sc_score': round(float(sc), 3),
        'votes':    f'{votes}/2',
    }


# ─── ADME Filter ───────────────────────────────────────────────────────────────

def adme_delta(smiles_orig: str, smiles_cand: str,
               delta_logp: float = 0.5,
               delta_mw:   float = 25.0) -> dict:
    """
    Check if a candidate preserves the ADME properties of the original molecule.
    Filters out bioisosteres that destroy drug absorption/efficacy.

    Roadmap spec: |ΔLogP| < 0.5 and |ΔMW| < 25 Da (Lipinski-inspired).

    Returns:
        {
            'passes': bool,
            'delta_logp': float,
            'delta_mw':   float,
            'logp_orig':  float,
            'logp_cand':  float,
            'mw_orig':    float,
            'mw_cand':    float,
        }
    """
    mol_orig = Chem.MolFromSmiles(smiles_orig)
    mol_cand = Chem.MolFromSmiles(smiles_cand)
    if mol_orig is None or mol_cand is None:
        return {'passes': False}

    logp_orig = Descriptors.MolLogP(mol_orig)
    logp_cand = Descriptors.MolLogP(mol_cand)
    mw_orig   = Descriptors.MolWt(mol_orig)
    mw_cand   = Descriptors.MolWt(mol_cand)

    dl = abs(logp_cand - logp_orig)
    dm = abs(mw_cand - mw_orig)

    return {
        'passes':    bool(dl <= delta_logp and dm <= delta_mw),
        'delta_logp': round(dl, 3),
        'delta_mw':   round(dm, 2),
        'logp_orig':  round(logp_orig, 3),
        'logp_cand':  round(logp_cand, 3),
        'mw_orig':    round(mw_orig, 2),
        'mw_cand':    round(mw_cand, 2),
    }


# ─── Fragment Extraction from SHAP bits ────────────────────────────────────────

def extract_fragment_from_bit(smiles: str, bit_index: int,
                               radius: int = 2, n_bits: int = 4096) -> Optional[str]:
    """
    Map a Morgan fingerprint bit index back to the molecular substructure that
    produced it, returned as a canonical SMILES fragment.

    How it works:
      RDKit's GetMorganFingerprintAsBitVect with bitInfo records:
        bitInfo[bit] = [(atom_idx, radius), ...]
      We use the atom and its environment (at the given radius) to extract
      the substructure as SMILES via Chem.MolFragmentToSmiles.

    Args:
        smiles:    Input molecule SMILES.
        bit_index: The Morgan FP bit to decode (0–4095 for 4096-bit ECFP8).
        radius:    Morgan radius used during featurization (default=2, ECFP4).
        n_bits:    Fingerprint size (default=4096).

    Returns:
        SMILES string of the fragment, or None if the bit is not set.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=bit_info
    )

    if bit_index not in bit_info:
        return None  # This bit is not set in this molecule

    # Take the first environment that produced this bit
    atom_idx, env_radius = bit_info[bit_index][0]

    # Extract the atom environment at the given radius
    env_atoms = Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, atom_idx)
    if not env_atoms and env_radius > 0:
        env_atoms = Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius - 1, atom_idx)

    if not env_atoms:
        # Radius 0 = just the atom itself
        atom_sym = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        return atom_sym

    atom_map = {}
    for bond_idx in env_atoms:
        bond = mol.GetBondWithIdx(bond_idx)
        atom_map[bond.GetBeginAtomIdx()] = -1
        atom_map[bond.GetEndAtomIdx()]   = -1
    atom_map[atom_idx] = -1

    frag_smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=list(atom_map.keys()),
        bondsToUse=list(env_atoms),
        canonical=True,
    )
    return frag_smiles if frag_smiles else None


# ─── ChEMBL Cache Builder (run offline, once) ──────────────────────────────────

def build_chembl_cache(fragment_smiles_list: list,
                       cache_path: str = '/workspace/data/chembl_cache.pkl',
                       max_per_fragment: int = 50) -> dict:
    """
    OFFLINE SCRIPT — run once to populate the local ChEMBL cache.
    Do NOT call this inside the prediction pipeline (rate-limited).

    Queries ChEMBL for molecules containing each fragment SMILES,
    caps at max_per_fragment results per fragment, saves as a pickle.

    Args:
        fragment_smiles_list: List of SMILES fragment strings to query.
        cache_path:           Where to save the cache.
        max_per_fragment:     Cap per fragment to avoid rate-limit hangs.

    Returns:
        The cache dict (also written to disk).
    """
    from chembl_webresource_client.new_client import new_client

    cache = {}
    molecule_client = new_client.molecule

    for frag in fragment_smiles_list:
        print(f'  Querying ChEMBL for fragment: {frag} ...')
        try:
            results = molecule_client.filter(
                molecule_structures__canonical_smiles__contains=frag
            ).only(['molecule_chembl_id', 'molecule_structures', 'molecule_properties'])

            cache[frag] = []
            for r in results[:max_per_fragment]:
                structs = r.get('molecule_structures') or {}
                props   = r.get('molecule_properties') or {}
                smiles  = structs.get('canonical_smiles')
                if smiles and Chem.MolFromSmiles(smiles) is not None:
                    cache[frag].append({
                        'chembl_id': r['molecule_chembl_id'],
                        'smiles':    smiles,
                        'mw':        props.get('full_mwt'),
                        'logp':      props.get('alogp'),
                    })
            print(f'    → {len(cache[frag])} valid molecules')
        except Exception as e:
            print(f'    → ERROR: {e}')
            cache[frag] = []

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f'\nChEMBL cache saved → {cache_path}')
    return cache


def load_chembl_cache(cache_path: str = '/workspace/data/chembl_cache.pkl') -> dict:
    """Load pre-built ChEMBL cache. Returns empty dict if not found."""
    if not os.path.exists(cache_path):
        warnings.warn(f'ChEMBL cache not found at {cache_path}. Run build_chembl_cache() first.')
        return {}
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def query_chembl_cache(cache: dict, fragment_smiles: str, cache_path: str = '/workspace/data/chembl_cache.pkl', max_live_results: int = 15) -> list:
    """
    Look up bioisostere candidates. First checks the local cache.
    If missing, performs a LIVE query to the ChEMBL API, updates the cache,
    and saves it back to disk so future lookups are instant.

    Returns:
        List of candidate dicts: [{'chembl_id', 'smiles', 'mw', 'logp'}, ...]
    """
    # 1. Exact Match
    if fragment_smiles in cache and len(cache[fragment_smiles]) > 0:
        return cache[fragment_smiles]

    # 2. Partial Match Fallback
    for cached_frag, candidates in cache.items():
        if len(candidates) > 0 and (fragment_smiles in cached_frag or cached_frag in fragment_smiles):
            return candidates

    # 3. LIVE API FALLBACK (The WOW Factor)
    print(f"  [!] Cache miss for {fragment_smiles}. Querying ChEMBL API live...")
    from chembl_webresource_client.new_client import new_client
    molecule_client = new_client.molecule
    
    try:
        results = molecule_client.filter(
            molecule_structures__canonical_smiles__contains=fragment_smiles
        ).only(['molecule_chembl_id', 'molecule_structures', 'molecule_properties'])

        candidates = []
        for r in results[:max_live_results]:
            structs = r.get('molecule_structures') or {}
            props   = r.get('molecule_properties') or {}
            smiles  = structs.get('canonical_smiles')
            if smiles and Chem.MolFromSmiles(smiles) is not None:
                candidates.append({
                    'chembl_id': r['molecule_chembl_id'],
                    'smiles':    smiles,
                    'mw':        props.get('full_mwt'),
                    'logp':      props.get('alogp'),
                })
        
        # Update memory and disk
        print(f"    → Found {len(candidates)} live candidates. Updating cache...")
        cache[fragment_smiles] = candidates
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"    → Warning: Could not save cache to disk: {e}")
            
        return candidates

    except Exception as e:
        print(f"    → Live ChEMBL API error: {e}")
        cache[fragment_smiles] = []
        return []
