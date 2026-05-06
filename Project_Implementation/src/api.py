import os
import sys
import json
import time
import asyncio
import numpy as np
from queue import Queue, Empty
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Ensure src is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.train import get_device
from src.explain import load_pipeline_from_artifact
from src.bioisostere import load_chembl_cache
from src.prescription_pipeline import run_prescription_pipeline

# ── New framework imports ──────────────────────────────────────────────────────
from src.stbi          import lookup_stbi
from src.constellations import classify_molecule
from src.escape_path   import compute_escape_pressures, summarize_escape_pressures
from src.prescription_pipeline import predict_single

# ─── Global State ─────────────────────────────────────────────────────────────

# We load ML models and caches ONCE at application startup.
ml_state = {}

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# Known Tox21 class imbalance ratios (negative:positive, from training analysis)
ENDPOINT_IMBALANCE = {
    'NR-AR':        16.0, 'NR-AR-LBD':    49.0, 'NR-AhR':       8.0,
    'NR-Aromatase': 20.0, 'NR-ER':         9.0, 'NR-ER-LBD':   22.0,
    'NR-PPAR-gamma':35.0, 'SR-ARE':        5.0, 'SR-ATAD5':    16.0,
    'SR-HSE':       13.0, 'SR-MMP':        5.0, 'SR-p53':       8.0,
}

# ─── Decision Trace Global ────────────────────────────────────────────────────
# Stores the structured audit log from the most recent /analyze call.
_last_trace: dict = {}

# ─── Pipeline Log Infrastructure ──────────────────────────────────────────────
# Thread-safe queue: analyze_molecule() puts log dicts here;
# /logs/stream SSE endpoint drains it to the browser.
_LOG_QUEUE: Queue = Queue(maxsize=500)
_pipeline_start_time: float = 0.0

LOG_ICONS = {
    'INFO':    'ℹ️',
    'SUCCESS': '✅',
    'WARN':    '⚠️',
    'ERROR':   '❌',
    'MODEL':   '🧠',
    'DATA':    '📊',
    'SHAP':    '🔭',
    'FRAG':    '🧲',
    'CHEM':    '🗂️',
    'PARETO':  '⚖️',
    'STBI':    '📈',
    'CONST':   '🌌',
    'MTEP':    '🗺️',
    'DONE':    '🏁',
    'START':   '🚀',
}

def pipeline_log(message: str, level: str = 'INFO', icon_key: str = 'INFO'):
    """Push a structured log event onto the SSE queue."""
    elapsed = round(time.time() - _pipeline_start_time, 3) if _pipeline_start_time else 0.0
    icon    = LOG_ICONS.get(icon_key, '•')
    entry   = {'t': elapsed, 'level': level, 'icon': icon, 'msg': message}
    try:
        _LOG_QUEUE.put_nowait(entry)
    except Exception:
        pass  # queue full — drop

# ─── Lifespan (Startup / Shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting ToxNet Prescription Engine API...")
    
    # 1. Device
    device = get_device()
    ml_state['device'] = device
    
    # 2. Model Pipeline
    model_path = os.path.join(PROJECT_ROOT, 'models', 'model_artifact.pkl')
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model artifact not found at {model_path}. Train model first.")
    
    print(f"Loading model to {device}...")
    pipeline, artifact = load_pipeline_from_artifact(model_path, device)
    ml_state['pipeline'] = pipeline
    ml_state['thresholds'] = artifact.get('thresholds', {})
    ml_state['fp_radius'] = artifact.get('fp_radius', 2)
    ml_state['fp_n_bits'] = artifact.get('fp_n_bits', 4096)
    
    # 3. Training Data (for SHAP background and OOD detection)
    train_data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'splits', 'X_fp_train.npy')
    if not os.path.exists(train_data_path):
        raise RuntimeError(f"Training data not found at {train_data_path}. Needed for OOD/SHAP.")
    
    print("Loading training fingerprints...")
    X_fp_train = np.load(train_data_path)
    ml_state['X_fp_train'] = X_fp_train
    # Use a small background sample for SHAP for API speed
    ml_state['X_train_bg'] = X_fp_train[:200]
    
    # 4. ChEMBL Cache
    print("Loading ChEMBL bioisostere cache...")
    chembl_cache_path = os.path.join(PROJECT_ROOT, 'data', 'chembl_cache.pkl')
    chembl_cache = load_chembl_cache(chembl_cache_path)
    if not chembl_cache:
        print("⚠️ WARNING: ChEMBL cache is empty. Bioisostere suggestions will fail.")
    ml_state['chembl_cache'] = chembl_cache

    # 5. STBI Artifact
    stbi_path = os.path.join(PROJECT_ROOT, 'data', 'stbi_artifact.pkl')
    if os.path.exists(stbi_path):
        import pickle
        with open(stbi_path, 'rb') as f:
            ml_state['stbi_artifact'] = pickle.load(f)
        print(f"✅ STBI artifact loaded ({len(ml_state['stbi_artifact'])} scaffolds).")
    else:
        ml_state['stbi_artifact'] = {}
        print("⚠️  STBI artifact not found — will compute on-the-fly.")

    # 5b. Load training DataFrame for on-the-fly STBI (unseen scaffolds)
    train_csv_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'tox21_cleaned.csv')
    if os.path.exists(train_csv_path):
        import pandas as pd
        ml_state['train_df'] = pd.read_csv(train_csv_path)
        print(f"✅ Training DataFrame loaded ({len(ml_state['train_df'])} rows) for live STBI.")
    else:
        ml_state['train_df'] = None
        print("⚠️  tox21_cleaned.csv not found — on-the-fly STBI unavailable.")

    # 6. Constellation Artifact
    const_path = os.path.join(PROJECT_ROOT, 'data', 'constellation_artifact.pkl')
    if os.path.exists(const_path):
        import pickle
        with open(const_path, 'rb') as f:
            ml_state['constellation_artifact'] = pickle.load(f)
        n_c = ml_state['constellation_artifact']['n_constellations']
        print(f"✅ Constellation artifact loaded ({n_c} constellations).")
    else:
        ml_state['constellation_artifact'] = None
        print("⚠️  Constellation artifact not found. Run scripts/precompute_frameworks.py first.")
    
    print("✅ API is ready to receive requests.")
    yield
    
    print("🛑 Shutting down API. Cleaning up ML resources...")
    ml_state.clear()

# ─── FastAPI App Initialization ───────────────────────────────────────────────

app = FastAPI(
    title="ToxNet Prescription Engine",
    description="Structural Optimization and Toxicity Prediction API",
    version="2.0.0",
    lifespan=lifespan
)

# Allow React frontend (e.g., localhost:5173, localhost:3000) to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the exact frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ──────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    smiles: str
    top_shap_bits: int = 3
    max_candidates: int = 15

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": 'pipeline' in ml_state}


# ─── SSE: Live Pipeline Log Stream ────────────────────────────────────────────

@app.get("/logs/stream")
async def stream_pipeline_logs():
    """
    Server-Sent Events endpoint.  The frontend connects here during analysis
    and receives structured log lines as the pipeline executes.
    """
    async def event_generator():
        # Drain any stale entries first
        while not _LOG_QUEUE.empty():
            try: _LOG_QUEUE.get_nowait()
            except Empty: break

        # Stream until DONE event is sent, then keep alive for 2s
        done_at = None
        while True:
            try:
                entry = _LOG_QUEUE.get_nowait()
                yield f"data: {json.dumps(entry)}\n\n"
                if entry.get('level') == 'DONE':
                    done_at = time.time()
            except Empty:
                if done_at and time.time() - done_at > 2.0:
                    break
                yield ": keepalive\n\n"
                await asyncio.sleep(0.08)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "online", "message": "ToxNet Prescription Engine API v2.0 — now with STBI, Constellations, and MTEP."}

@app.get("/trace")
def get_decision_trace():
    """
    Returns the full structured decision trace from the most recent /analyze call.
    Includes every filter, score, pass/fail decision, and classification made.
    """
    if not _last_trace:
        return {"error": "No analysis has been run yet. Submit a molecule first."}
    return _last_trace

@app.post("/analyze")
def analyze_molecule(req: AnalysisRequest):
    """
    Run the full prescription pipeline + three groundbreaking frameworks.
    Emits structured log events to _LOG_QUEUE for SSE streaming.
    """
    global _pipeline_start_time
    _pipeline_start_time = time.time()

    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES string cannot be empty.")

    pipeline_log(f"Received SMILES: {smiles[:60]}{'...' if len(smiles) > 60 else ''}", icon_key='START')
    pipeline_log(f"Pipeline config: SHAP bits={req.top_shap_bits}, max candidates={req.max_candidates}", icon_key='DATA')

    try:
        # ── Parsing & validation ──────────────────────────────────────────────
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw   = round(Descriptors.MolWt(mol), 2)
            hba  = rdMolDescriptors.CalcNumHBA(mol)
            hbd  = rdMolDescriptors.CalcNumHBD(mol)
            rings = rdMolDescriptors.CalcNumRings(mol)
            n_atoms = mol.GetNumAtoms()
            pipeline_log(f"Molecule parsed — {n_atoms} heavy atoms, MW={mw}, HBA={hba}, HBD={hbd}, rings={rings}", level='SUCCESS', icon_key='DATA')
        else:
            pipeline_log("RDKit parse warning — falling back to pipeline validator", level='WARN', icon_key='WARN')

        pipeline_log(f"Computing {ml_state['fp_n_bits']}-bit Morgan fingerprint (radius={ml_state['fp_radius']})...", icon_key='DATA')

        # ── Core prescription pipeline ────────────────────────────────────────
        pipeline_log("ToxNetLite → Pass 1 inference (lightweight embedding)...", icon_key='MODEL')
        result = run_prescription_pipeline(
            smiles=smiles,
            pipeline=ml_state['pipeline'],
            thresholds=ml_state['thresholds'],
            target_cols=TARGET_COLS,
            X_train_bg=ml_state['X_train_bg'],
            X_fp_train=ml_state['X_fp_train'],
            device=ml_state['device'],
            chembl_cache=ml_state['chembl_cache'],
            fp_radius=ml_state['fp_radius'],
            fp_n_bits=ml_state['fp_n_bits'],
            top_shap_bits=req.top_shap_bits,
            max_candidates=req.max_candidates,
            shap_bg_size=100
        )

        # Log prediction outcome
        n_flagged = result.get('prediction', {}).get('n_flagged', 0)
        mean_prob = result.get('prediction', {}).get('mean_prob', 0)
        flags     = result.get('prediction', {}).get('flags', {})
        flagged_eps = [ep for ep, v in flags.items() if v]

        pipeline_log(f"ToxNet → Pass 2 complete — 12 endpoints evaluated, mean prob={mean_prob:.3f}", level='SUCCESS', icon_key='MODEL')

        if n_flagged == 0:
            pipeline_log("No endpoints flagged — molecule appears safe across all 12 assays", level='SUCCESS', icon_key='SUCCESS')
        else:
            pipeline_log(f"{n_flagged} endpoint(s) flagged: {', '.join(flagged_eps[:6])}{'...' if len(flagged_eps) > 6 else ''}", level='WARN', icon_key='WARN')

        # OOD check
        ood = result.get('prediction', {}).get('ood_warning')
        if ood:
            pipeline_log(f"OOD warning: molecule is {ood} from training distribution — confidence reduced", level='WARN', icon_key='WARN')
        else:
            pipeline_log("OOD check passed — molecule is within training applicability domain", icon_key='SUCCESS')

        # SHAP
        n_shap = len(result.get('shap_fragments', []))
        top_frag = result.get('primary_toxic_fragment') or '(none)'
        pipeline_log(f"SHAP DeepExplainer — {req.top_shap_bits} top bits extracted from 4096-bit space", icon_key='SHAP')
        pipeline_log(f"Primary toxic fragment: {top_frag[:50]}", icon_key='FRAG')

        # Candidates
        n_cands = result.get('n_candidates_found', 0)
        n_after = result.get('n_after_filter', 0)
        pipeline_log(f"ChEMBL cache lookup — {n_cands} bioisostere candidates retrieved", icon_key='CHEM')
        if n_after > 0:
            pipeline_log(f"Pareto dominance ranking — {n_after} candidates survived tox×SA×SC filter", level='SUCCESS', icon_key='PARETO')
            top_c = result.get('pareto_candidates', [{}])[0]
            if top_c:
                tc_smi = top_c.get('smiles', '')[:40]
                pipeline_log(f"Rank-1 candidate: {tc_smi} (SA={top_c.get('sa_score','?')}, SC={top_c.get('sc_score','?')})", level='SUCCESS', icon_key='PARETO')
        else:
            pipeline_log("No Pareto-optimal candidates found — all ChEMBL hits dominated or out-of-domain", level='WARN', icon_key='WARN')

        if result['pipeline_status'] == 'error:invalid_smiles':
            raise HTTPException(status_code=400, detail="Invalid SMILES string provided.")

        # ── Framework 1: STBI ─────────────────────────────────────────────────
        pipeline_log("Computing Scaffold Toxicity Brittleness Index (STBI)...", icon_key='STBI')
        stbi_artifact = ml_state.get('stbi_artifact') or {}
        try:
            stbi_result = lookup_stbi(smiles, stbi_artifact)

            # If scaffold not in pre-built artifact, compute live from training data
            if stbi_result['assessment'] == 'UNSEEN' and ml_state.get('train_df') is not None:
                from src.stbi import compute_stbi_for_endpoint, _get_scaffold
                scaffold = stbi_result.get('scaffold')
                if scaffold:
                    train_df = ml_state['train_df']
                    # Only rows sharing this exact Murcko scaffold
                    live_ep_scores = {}
                    for ep in TARGET_COLS:
                        if ep not in train_df.columns:
                            continue
                        try:
                            ep_scores = compute_stbi_for_endpoint(train_df, ep, fp_radius=2, fp_n_bits=2048)
                            if scaffold in ep_scores:
                                live_ep_scores[ep] = ep_scores[scaffold]
                        except Exception:
                            pass

                    if live_ep_scores:
                        max_s  = max(live_ep_scores.values())
                        mean_s = float(sum(live_ep_scores.values()) / len(live_ep_scores))
                        # Re-run lookup_stbi with an on-the-fly mini-artifact
                        mini_artifact = {scaffold: {**live_ep_scores, 'max_stbi': round(max_s, 4), 'mean_stbi': round(mean_s, 4)}}
                        stbi_result = lookup_stbi(smiles, mini_artifact)
                        stbi_result['live_computed'] = True
                    else:
                        stbi_result['message'] = (
                            'This scaffold has no Tox21 training members with both toxic AND safe '
                            'labels — STBI requires both classes to measure brittleness. '
                            'Predictions are based solely on the neural network.'
                        )

            result['stbi'] = stbi_result
            stbi_score = stbi_result.get('stbi')
            stbi_asmt  = stbi_result.get('assessment', '?')
            if stbi_score is not None:
                pipeline_log(f"STBI result: {stbi_asmt} (score={stbi_score:.3f}) — {'live computed' if stbi_result.get('live_computed') else 'from artifact'}", level='SUCCESS', icon_key='STBI')
            else:
                pipeline_log(f"STBI: {stbi_asmt} — scaffold not in training set (STBI requires mixed-class scaffold members)", level='WARN', icon_key='STBI')
        except Exception as e:
            result['stbi'] = {'assessment': 'ERROR', 'message': str(e), 'stbi': None}

        # ── Framework 2: Toxicity Constellations ──────────────────────────────
        pipeline_log("Classifying toxicity constellation (12D probability clustering)...", icon_key='CONST')
        if ml_state.get('constellation_artifact') is not None:
            try:
                pred  = result.get('prediction', {})
                probs = pred.get('probabilities', {})
                if probs:
                    prob_vector = np.array(
                        [float(probs.get(ep, 0.0)) for ep in TARGET_COLS],
                        dtype=np.float32,
                    )
                    c = classify_molecule(prob_vector, ml_state['constellation_artifact'])
                    result['constellation'] = c
                    pipeline_log(
                        f"Constellation: {c.get('constellation_name', '?')} — "
                        f"{c.get('mechanism_hint', '?')[:60]} (proximity={c.get('proximity_label', '?')})",
                        level='SUCCESS', icon_key='CONST'
                    )
                else:
                    result['constellation'] = None
            except Exception as e:
                result['constellation'] = {'error': str(e)}
                pipeline_log(f"Constellation error: {e}", level='ERROR', icon_key='CONST')
        else:
            result['constellation'] = {
                'constellation_name': 'UNAVAILABLE',
                'mechanism_hint': 'Run scripts/precompute_frameworks.py to enable Constellations.',
            }

        # ── Framework 3: MTEP (Minimum Toxicity Escape Path) ──────────────────
        pipeline_log("Computing Minimum Toxicity Escape Path (gradient-based MTEP)...", icon_key='MTEP')
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs
            import torch as _torch

            _mol = Chem.MolFromSmiles(smiles)
            if _mol is not None:
                _fp_n_bits = ml_state['fp_n_bits']
                _fp_radius = ml_state['fp_radius']
                _fp_arr = np.zeros(_fp_n_bits, dtype=np.float32)
                _fp = AllChem.GetMorganFingerprintAsBitVect(_mol, radius=_fp_radius, nBits=_fp_n_bits)
                DataStructs.ConvertToNumpyArray(_fp, _fp_arr)

                _const_art = ml_state.get('constellation_artifact') or {}
                _easy_thr  = _const_art.get('mtep_easy_thr', 0.15)
                _hard_thr  = _const_art.get('mtep_hard_thr', 0.06)

                ep_result = compute_escape_pressures(
                    fp_arr=_fp_arr,
                    pipeline=ml_state['pipeline'],
                    thresholds=ml_state['thresholds'],
                    target_cols=TARGET_COLS,
                    device=ml_state['device'],
                    easy_thr=_easy_thr,
                    hard_thr=_hard_thr,
                )
                summary = summarize_escape_pressures(ep_result)
                result['escape_path'] = {'per_endpoint': ep_result, 'summary': summary}

                _s        = summary
                easy_n    = _s.get('n_easy', 0)
                hard_n    = _s.get('n_hard', 0)
                trapped_n = _s.get('n_trapped', 0)
                prognosis = _s.get('overall_prognosis', '?')
                pipeline_log(f"MTEP: prognosis={prognosis} — {easy_n} EASY / {hard_n} HARD / {trapped_n} TRAPPED", level='SUCCESS', icon_key='MTEP')
            else:
                result['escape_path'] = None
                pipeline_log("MTEP skipped — molecule could not be parsed", level='WARN', icon_key='MTEP')
        except Exception as e:
            result['escape_path'] = {'error': str(e)}
            pipeline_log(f"MTEP error: {e}", level='ERROR', icon_key='MTEP')


        elapsed_total = round(time.time() - _pipeline_start_time, 3)
        pipeline_log(f"Pipeline complete — total time {elapsed_total}s", level='DONE', icon_key='DONE')

        # ── Build full Decision Trace ──────────────────────────────────────────
        global _last_trace
        pred    = result.get('prediction', {})
        probs   = pred.get('probabilities', {})
        flags   = pred.get('flags', {})
        thresholds = ml_state.get('thresholds', {})

        # Section 1 — Molecule properties
        try:
            from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
            _m = Chem.MolFromSmiles(smiles)
            mol_props_trace = {
                'smiles':     smiles,
                'mw':         round(Descriptors.MolWt(_m), 2),
                'logp':       round(Crippen.MolLogP(_m), 3),
                'hba':        rdMolDescriptors.CalcNumHBA(_m),
                'hbd':        rdMolDescriptors.CalcNumHBD(_m),
                'tpsa':       round(rdMolDescriptors.CalcTPSA(_m), 2),
                'rot_bonds':  rdMolDescriptors.CalcNumRotatableBonds(_m),
                'rings':      rdMolDescriptors.CalcNumRings(_m),
                'heavy_atoms':_m.GetNumAtoms(),
                'lipinski_ok': all([
                    Descriptors.MolWt(_m) <= 500,
                    Crippen.MolLogP(_m) <= 5,
                    rdMolDescriptors.CalcNumHBA(_m) <= 10,
                    rdMolDescriptors.CalcNumHBD(_m) <= 5,
                ]),
            }
        except Exception:
            mol_props_trace = {'smiles': smiles}

        # Section 2 — OOD assessment
        ood_trace = {
            'in_domain': not bool(pred.get('ood_warning')),
            'ood_warning': pred.get('ood_warning'),
            'message': 'Within training applicability domain' if not pred.get('ood_warning')
                       else f'Out-of-domain: {pred["ood_warning"]}',
        }

        # Section 3 — Per-endpoint predictions
        endpoint_trace = []
        for ep in TARGET_COLS:
            prob      = float(probs.get(ep, 0.0))
            threshold = float(thresholds.get(ep, 0.5))
            flagged   = bool(flags.get(ep, False))
            imbalance = ENDPOINT_IMBALANCE.get(ep, '?')
            endpoint_trace.append({
                'endpoint':  ep,
                'prob':      round(prob, 4),
                'threshold': round(threshold, 4),
                'flagged':   flagged,
                'imbalance': imbalance,
                'margin':    round(abs(prob - threshold), 4),
                'decision':  'TOXIC' if flagged else ('NEAR' if abs(prob - threshold) < 0.1 else 'SAFE'),
            })

        # Section 4 — SHAP attribution
        shap_trace = []
        for frag in (result.get('shap_fragments') or []):
            shap_trace.append({
                'bit':       frag.get('bit'),
                'fragment':  frag.get('fragment'),
                'importance':frag.get('importance'),
                'endpoints': frag.get('endpoint_drivers', []),
            })

        # Section 5 — Candidate pipeline
        candidates_trace = []
        for c in (result.get('pareto_candidates') or []):
            candidates_trace.append({
                'smiles':     c.get('smiles'),
                'chembl_id':  c.get('chembl_id'),
                'rank':       c.get('rank'),
                'sa_score':   c.get('sa_score'),
                'sc_score':   c.get('sc_score'),
                'synth_verdict': c.get('synth_verdict'),
                'mean_tox_delta': c.get('mean_tox_delta'),
                'adme_passes':c.get('adme_passes', True),
                'delta_logp': c.get('delta_logp'),
                'delta_mw':   c.get('delta_mw'),
                'pareto_dominated': c.get('rank', 1) > 3,
            })
        candidate_pipeline_trace = {
            'fragment_queried':  result.get('primary_toxic_fragment'),
            'chembl_found':      result.get('n_candidates_found', 0),
            'after_adme':        result.get('n_after_filter', 0),
            'final_candidates':  len(candidates_trace),
            'candidates':        candidates_trace[:15],
        }

        # Section 6 — STBI
        stbi_trace = {}
        if result.get('stbi'):
            s = result['stbi']
            stbi_trace = {
                'scaffold':      s.get('scaffold'),
                'assessment':    s.get('assessment'),
                'score':         s.get('stbi'),
                'live_computed': s.get('live_computed', False),
                'message':       s.get('message'),
                'endpoint_scores': s.get('endpoint_scores', {}),
            }

        # Section 7 — Constellation
        const_trace = {}
        if result.get('constellation') and isinstance(result['constellation'], dict):
            c = result['constellation']
            const_trace = {
                'name':            c.get('constellation_name'),
                'proximity':       c.get('proximity_label'),
                'distance':        round(float(c.get('distance') or 0), 3),
                'mechanism':       c.get('mechanism_hint'),
                'dominant_endpoints': c.get('dominant_endpoints', []),
                'centroid_profile':c.get('centroid_profile', {}),
            }

        # Section 8 — MTEP
        mtep_trace = []
        if result.get('escape_path') and isinstance(result['escape_path'], dict):
            per_ep = result['escape_path'].get('per_endpoint', {})
            summary_ep = result['escape_path'].get('summary', {})
            for ep, data in per_ep.items():
                if isinstance(data, dict):
                    mtep_trace.append({
                        'endpoint':      ep,
                        'probability':   round(float(data.get('probability') or 0), 4),
                        'flagged':       data.get('flagged', False),
                        'gradient_norm': round(float(data.get('gradient_norm') or 0), 5),
                        'difficulty':    data.get('difficulty', 'CLEAN'),
                        'top_escape_bits': data.get('top_escape_bits', [])[:3],
                    })
            mtep_summary_trace = {
                'prognosis':  summary_ep.get('overall_prognosis'),
                'n_easy':     summary_ep.get('n_easy', 0),
                'n_hard':     summary_ep.get('n_hard', 0),
                'n_trapped':  summary_ep.get('n_trapped', 0),
                'message':    summary_ep.get('prognosis_message'),
            }
        else:
            mtep_summary_trace = {}

        # ── Section 9 — Structural Alerts (expose in trace) ───────────────────────
        _raw_alerts = result.get('alerts', {})
        alerts_trace = {
            'pains_hit':       _raw_alerts.get('pains_hit', False),
            'brenk_hit':       _raw_alerts.get('brenk_hit', False),
            'alert_names':     _raw_alerts.get('alert_names', []),
            'shap_confidence': _raw_alerts.get('shap_confidence', 'MEDIUM'),
        }

        # ── Section 10 — Confidence Chain ─────────────────────────────────────────
        _steps = []
        _flagged_eps = [e for e in endpoint_trace if e['flagged']]
        _all_eps     = endpoint_trace

        # Step 1: Base ML score
        _base_prob = float(np.mean([e['prob'] for e in _flagged_eps])) if _flagged_eps else float(np.mean([e['prob'] for e in _all_eps]))
        _base_pct  = round(_base_prob * 100, 1)
        _running   = _base_pct
        _steps.append({
            'step': 1, 'icon': '🧠', 'name': 'ToxNet ML Model',
            'description': (f'Sigmoid probability from 2-pass ToxNetLite→ToxNet architecture (ECFP4, 4096-bit Morgan fingerprints). '
                            f'{"Flagged" if _flagged_eps else "All"} endpoint mean: {_base_pct}%.'),
            'delta': 0, 'running_total': round(_running, 1), 'direction': 'base',
        })

        # Step 2: Calibrated Thresholds (Mondrian CP)
        _mean_margin = float(np.mean([e['margin'] for e in _flagged_eps])) if _flagged_eps else 0.0
        if _mean_margin > 0.15:
            _thr_delta, _thr_dir = 9, 'positive'
            _thr_desc = f'Mean margin above F1-optimal threshold: {_mean_margin:.3f} — predictions are well-separated from decision boundary. High calibration confidence.'
        elif _mean_margin > 0.05:
            _thr_delta, _thr_dir = 4, 'positive'
            _thr_desc = f'Mean margin above threshold: {_mean_margin:.3f} — moderate separation from boundary. Good calibration confidence.'
        elif _mean_margin > 0.0:
            _thr_delta, _thr_dir = 1, 'positive'
            _thr_desc = f'Mean margin: {_mean_margin:.3f} — predictions near the calibrated boundary. Threshold uncertainty applies.'
        else:
            _thr_delta, _thr_dir = -5, 'negative'
            _thr_desc = 'No endpoints exceed calibrated F1-optimal thresholds — all predictions below decision boundary.'
        _running += _thr_delta
        _steps.append({
            'step': 2, 'icon': '📏', 'name': 'Calibrated Thresholds (Mondrian CP)',
            'description': _thr_desc,
            'delta': _thr_delta, 'running_total': round(_running, 1), 'direction': _thr_dir,
        })

        # Step 3: Class Imbalance Correction
        if _flagged_eps:
            _imb_vals = [e['imbalance'] for e in _flagged_eps if isinstance(e.get('imbalance'), (int, float))]
            _mean_imb = float(np.mean(_imb_vals)) if _imb_vals else 1.0
            if _mean_imb >= 20:
                _imb_delta, _imb_dir = 7, 'positive'
                _imb_desc = f'Mean training imbalance: 1:{_mean_imb:.0f} — sparse toxic class means any flagged prediction is high-precision evidence.'
            elif _mean_imb >= 10:
                _imb_delta, _imb_dir = 4, 'positive'
                _imb_desc = f'Mean training imbalance: 1:{_mean_imb:.0f} — moderate imbalance amplifies significance of flagged endpoints.'
            else:
                _imb_delta, _imb_dir = 2, 'positive'
                _imb_desc = f'Mean training imbalance: 1:{_mean_imb:.0f} — relatively balanced classes. Standard significance.'
        else:
            _imb_delta, _imb_dir = 0, 'neutral'
            _imb_desc = 'No endpoints flagged — imbalance weighting not applicable for this molecule.'
        _running += _imb_delta
        _steps.append({
            'step': 3, 'icon': '⚖️', 'name': 'Class Imbalance Correction',
            'description': _imb_desc,
            'delta': _imb_delta, 'running_total': round(_running, 1), 'direction': _imb_dir,
        })

        # Step 4: OOD Applicability Domain
        if ood_trace.get('in_domain', True):
            _ood_delta, _ood_dir = 8, 'positive'
            _ood_desc = 'Within training applicability domain (Tanimoto max_sim ≥ 0.4). Model is interpolating within known chemical space — high reliability.'
        else:
            _ood_delta, _ood_dir = -15, 'negative'
            _ood_desc = f'OUT-OF-DOMAIN: {ood_trace.get("ood_warning", "distance exceeds threshold")}. Model is extrapolating — treat all predictions with caution.'
        _running += _ood_delta
        _steps.append({
            'step': 4, 'icon': '🎯', 'name': 'OOD Applicability Domain',
            'description': _ood_desc,
            'delta': _ood_delta, 'running_total': round(_running, 1), 'direction': _ood_dir,
        })

        # Step 5: PAINS/Brenk Structural Alerts
        _shap_conf   = alerts_trace.get('shap_confidence', 'MEDIUM')
        _alert_names = alerts_trace.get('alert_names', [])
        if _shap_conf == 'HIGH':
            _alert_delta, _alert_dir = 6, 'positive'
            _alert_desc = 'No PAINS/Brenk structural alerts — molecule does not contain known reactive scaffolds. SHAP attributions are fully trusted.'
        elif _shap_conf == 'MEDIUM':
            _alert_delta, _alert_dir = 0, 'neutral'
            _alert_desc = f'1 structural alert ({_alert_names[0] if _alert_names else "unknown"}). SHAP attributions may partially overlap with reactive group. Moderate trust.'
        else:
            _alert_delta, _alert_dir = -8, 'negative'
            _alert_desc = f'{len(_alert_names)} structural alerts ({", ".join(_alert_names[:2])}). SHAP reliability is reduced — multiple reactive groups detected.'
        _running += _alert_delta
        _steps.append({
            'step': 5, 'icon': '🚨', 'name': 'PAINS/Brenk Structural Alerts',
            'description': _alert_desc,
            'delta': _alert_delta, 'running_total': round(_running, 1), 'direction': _alert_dir,
        })

        # Step 6: SHAP GradientExplainer
        if _shap_conf == 'HIGH':
            _shap_delta, _shap_dir = 5, 'positive'
            _shap_desc = 'SHAP GradientExplainer attributions are chemically consistent — top Morgan bits correspond to structurally meaningful fragments.'
        elif _shap_conf == 'MEDIUM':
            _shap_delta, _shap_dir = 2, 'positive'
            _shap_desc = 'SHAP attributions partially validated. Some bits may overlap with reactive groups, but top-fragment identification is still informative.'
        else:
            _shap_delta, _shap_dir = -5, 'negative'
            _shap_desc = 'SHAP attributions less reliable — multiple structural alerts may confound bit-level explanations. Fragment suggestions are indicative only.'
        _running += _shap_delta
        _steps.append({
            'step': 6, 'icon': '🔭', 'name': 'SHAP GradientExplainer',
            'description': _shap_desc,
            'delta': _shap_delta, 'running_total': round(_running, 1), 'direction': _shap_dir,
        })

        # Step 7: STBI Scaffold Brittleness
        _stbi_asmt  = stbi_trace.get('assessment', 'UNSEEN')
        _stbi_score = stbi_trace.get('score')
        _stbi_score_str = f'{_stbi_score:.4f}' if _stbi_score is not None else 'N/A'
        if _stbi_asmt == 'STABLE':
            _stbi_delta, _stbi_dir = 5, 'positive'
            _stbi_desc = f'STBI = {_stbi_score_str} — scaffold boundary is stable. Toxicity prediction is consistent across scaffold members. High structural confidence.'
        elif _stbi_asmt == 'BRITTLE':
            _stbi_delta, _stbi_dir = -8, 'negative'
            _stbi_desc = f'STBI = {_stbi_score_str} — scaffold is toxicity-brittle. Minor structural changes dramatically alter the prediction. Confidence reduced.'
        else:
            _stbi_delta, _stbi_dir = 0, 'neutral'
            _stbi_desc = f'STBI assessment: {_stbi_asmt} — scaffold not seen in training or insufficient mixed-class members. No adjustment made.'
        _running += _stbi_delta
        _steps.append({
            'step': 7, 'icon': '📈', 'name': 'Scaffold Brittleness (STBI)',
            'description': _stbi_desc,
            'delta': _stbi_delta, 'running_total': round(_running, 1), 'direction': _stbi_dir,
        })

        # Step 8: Toxicity Constellation
        _const_name = const_trace.get('name')
        _const_prox = const_trace.get('proximity', 'FRINGE')
        if _const_name and _const_name not in ('UNAVAILABLE', 'ERROR', None):
            if _const_prox == 'CORE':
                _const_delta, _const_dir = 8, 'positive'
                _const_desc = f'CORE proximity to "{_const_name}" constellation — molecule is central to this toxicity cluster. Strong mechanism corroboration.'
            elif _const_prox == 'PERIPHERAL':
                _const_delta, _const_dir = 4, 'positive'
                _const_desc = f'PERIPHERAL proximity to "{_const_name}" constellation — molecule is near but not central to this cluster.'
            else:
                _const_delta, _const_dir = 1, 'positive'
                _const_desc = f'FRINGE proximity to "{_const_name}" — weak cluster membership. Mechanism hint: {str(const_trace.get("mechanism", ""))[:70]}.'
        else:
            _const_delta, _const_dir = 0, 'neutral'
            _const_desc = 'Constellation artifact unavailable. Run precompute_frameworks.py to enable this corroboration layer.'
        _running += _const_delta
        _steps.append({
            'step': 8, 'icon': '🌌', 'name': 'Toxicity Constellation',
            'description': _const_desc,
            'delta': _const_delta, 'running_total': round(_running, 1), 'direction': _const_dir,
        })

        # Step 9: MTEP Escape Gradient
        _mtep_prog = mtep_summary_trace.get('prognosis')
        if _mtep_prog == 'TRAPPED':
            _mtep_delta, _mtep_dir = 10, 'positive'
            _mtep_desc = 'TRAPPED endpoints — high gradient norms confirm molecule is deep in the toxic energy basin. Structural escape is infeasible. Maximum confidence.'
        elif _mtep_prog == 'HARD':
            _mtep_delta, _mtep_dir = 5, 'positive'
            _mtep_desc = 'HARD escape paths — significant structural changes needed to reduce toxicity. Strong confidence in toxic classification.'
        elif _mtep_prog == 'MODERATE':
            _mtep_delta, _mtep_dir = 0, 'neutral'
            _mtep_desc = 'MODERATE escape paths — mixed gradient norms. Some endpoints addressable with targeted modifications.'
        elif _mtep_prog == 'EASY':
            _mtep_delta, _mtep_dir = -5, 'negative'
            _mtep_desc = 'EASY escape paths — low gradient norms suggest minor changes could reduce toxicity. Slightly reduces confidence in irreversible toxic character.'
        elif _mtep_prog == 'CLEAN':
            _mtep_delta, _mtep_dir = -5, 'negative'
            _mtep_desc = 'CLEAN — no toxic endpoints. MTEP confirms molecule is not in a toxic energy basin.'
        else:
            _mtep_delta, _mtep_dir = 0, 'neutral'
            _mtep_desc = 'MTEP data unavailable or inconclusive.'
        _running += _mtep_delta
        _steps.append({
            'step': 9, 'icon': '🗺️', 'name': 'MTEP Escape Gradient',
            'description': _mtep_desc,
            'delta': _mtep_delta, 'running_total': round(_running, 1), 'direction': _mtep_dir,
        })

        # Final verdict
        _final_conf = max(0.0, min(100.0, round(_running, 1)))
        if _final_conf >= 75:
            _verdict, _verdict_level = 'HIGH CONFIDENCE TOXIC', 'danger'
        elif _final_conf >= 55:
            _verdict, _verdict_level = 'MODERATE CONFIDENCE TOXIC', 'warn'
        elif _final_conf >= 40:
            _verdict, _verdict_level = 'UNCERTAIN — BORDERLINE', 'neutral'
        elif _final_conf >= 25:
            _verdict, _verdict_level = 'MODERATE CONFIDENCE SAFE', 'good'
        else:
            _verdict, _verdict_level = 'HIGH CONFIDENCE SAFE', 'safe'

        confidence_chain = {
            'steps':            _steps,
            'final_confidence': _final_conf,
            'verdict':          _verdict,
            'verdict_level':    _verdict_level,
        }

        _last_trace = {
            'smiles':           smiles,
            'elapsed_s':        elapsed_total,
            'timestamp':        time.strftime('%Y-%m-%d %H:%M:%S'),
            'mol_properties':   mol_props_trace,
            'ood':              ood_trace,
            'endpoints':        endpoint_trace,
            'shap':             shap_trace,
            'candidates':       candidate_pipeline_trace,
            'stbi':             stbi_trace,
            'constellation':    const_trace,
            'mtep':             mtep_trace,
            'mtep_summary':     mtep_summary_trace,
            'alerts':           alerts_trace,
            'confidence_chain': confidence_chain,
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── /mol3d: 3D Conformation + SHAP Atom Coloring ────────────────────────────

class ShapBitInput(BaseModel):
    bit: int
    importance: float

class Mol3DRequest(BaseModel):
    smiles: str
    shap_bits: list[ShapBitInput] = []   # Pre-computed from /analyze — avoids re-running SHAP

@app.post("/mol3d")
def generate_3d_molecule(req: Mol3DRequest):
    """
    Generate a 3D conformation of the molecule and map SHAP bit scores onto atoms.

    The shap_bits list (pre-computed by /analyze) maps directly:
      bit_idx → importance → all atoms in that Morgan environment get the score.

    This avoids re-running DeepExplainer here, which was failing silently.
    """
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES cannot be empty.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string.")

    # ── Step 1: Generate 3D conformer ─────────────────────────────────────────
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.enforceChirality = True
    embed_result = AllChem.EmbedMolecule(mol_h, params)
    if embed_result == -1:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        pass

    mol_3d = Chem.RemoveHs(mol_h)
    sdf_block = Chem.MolToMolBlock(mol_3d)
    n_atoms = mol_3d.GetNumAtoms()

    # ── Step 2: Build bitInfo dict ────────────────────────────────────────────
    fp_n_bits = ml_state['fp_n_bits']
    fp_radius  = ml_state['fp_radius']
    bit_info: dict = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=fp_radius, nBits=fp_n_bits, bitInfo=bit_info
    )
    fp_arr = np.zeros(fp_n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    # ── Step 3: Build agg_shap from pre-computed bits OR fallback to SHAP ─────
    agg_shap = np.zeros(fp_n_bits, dtype=np.float32)

    if req.shap_bits:
        # Use pre-computed values from /analyze — reliable, no re-run needed
        for item in req.shap_bits:
            if 0 <= item.bit < fp_n_bits:
                agg_shap[item.bit] = max(0.0, float(item.importance))
        # Normalise
        max_val = agg_shap.max()
        if max_val > 0:
            agg_shap /= max_val
        print(f"[mol3d] Using {len(req.shap_bits)} pre-computed SHAP bits. "
              f"Max score: {agg_shap.max():.4f}")
    else:
        # Fallback: run DeepExplainer ourselves (may be slower)
        try:
            import shap as shap_lib
            pipeline  = ml_state['pipeline']
            X_bg      = ml_state['X_train_bg'].astype(np.float32)
            bg_tensor    = torch.from_numpy(X_bg).to(ml_state['device'])
            query_tensor = torch.from_numpy(fp_arr[np.newaxis]).to(ml_state['device'])
            explainer = shap_lib.DeepExplainer(pipeline, bg_tensor)
            shap_vals = explainer.shap_values(query_tensor)
            if isinstance(shap_vals, list):
                raw = np.sum([np.abs(sv[0]) for sv in shap_vals], axis=0)
            else:
                raw = np.abs(shap_vals[0])
            if raw.max() > 0:
                agg_shap = (raw / raw.max()).astype(np.float32)
            print(f"[mol3d] SHAP fallback succeeded. Max: {agg_shap.max():.4f}")
        except Exception as shap_err:
            print(f"[mol3d] SHAP fallback also failed: {shap_err}")

    # ── Step 4: Map bit scores → per-atom scores via bitInfo ──────────────────
    atom_shap      = np.zeros(n_atoms, dtype=np.float32)
    atom_hit_count = np.zeros(n_atoms, dtype=np.int32)

    for bit_idx, environments in bit_info.items():
        bit_score = float(agg_shap[bit_idx]) if bit_idx < fp_n_bits else 0.0
        if bit_score < 1e-6:
            continue  # Skip bits with no SHAP contribution
        for (center_atom, radius) in environments:
            env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol_3d, radius, center_atom)
            if env_bonds:
                involved = set()
                for bond_idx in env_bonds:
                    bond = mol_3d.GetBondWithIdx(bond_idx)
                    involved.add(bond.GetBeginAtomIdx())
                    involved.add(bond.GetEndAtomIdx())
                involved.add(center_atom)
            else:
                involved = {center_atom}
            for a in involved:
                if 0 <= a < n_atoms:
                    atom_shap[a]      += bit_score
                    atom_hit_count[a] += 1

    # Average overlapping contributions
    mask = atom_hit_count > 0
    atom_shap[mask] /= atom_hit_count[mask]

    # Normalise final atom scores to [0, 1]
    if atom_shap.max() > 0:
        atom_shap = atom_shap / atom_shap.max()

    print(f"[mol3d] Atom SHAP stats: max={atom_shap.max():.4f} "
          f"nonzero={int((atom_shap > 0).sum())}/{n_atoms}")

    # ── Step 5: Endpoint predictions for is_toxic flag ────────────────────────
    thresholds = ml_state['thresholds']
    pipeline   = ml_state['pipeline']
    pipeline.eval()
    with torch.no_grad():
        x_t    = torch.from_numpy(fp_arr[np.newaxis]).to(ml_state['device'])
        probs_t = pipeline(x_t).cpu().numpy()[0]

    flagged_count    = sum(1 for i, ep in enumerate(TARGET_COLS) if probs_t[i] >= thresholds.get(ep, 0.5))
    molecule_is_toxic = flagged_count > 0

    # Top 25% of non-zero atoms → is_toxic
    nonzero_scores = atom_shap[atom_shap > 0]
    toxic_threshold = float(np.percentile(nonzero_scores, 75)) if len(nonzero_scores) > 0 else 0.5

    atom_colors = [
        {
            "atom_idx":  int(i),
            "shap_score": round(float(atom_shap[i]), 4),
            "is_toxic":   bool(molecule_is_toxic and atom_shap[i] >= toxic_threshold),
        }
        for i in range(n_atoms)
    ]
    n_toxic_atoms = sum(1 for a in atom_colors if a["is_toxic"])

    # ── Step 6: Top fragment SMILES ───────────────────────────────────────────
    top_fragment = None
    if agg_shap.max() > 0:
        top_bit = int(np.argmax(agg_shap))
        if top_bit in bit_info:
            center_atom, radius_r = bit_info[top_bit][0]
            try:
                env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol_3d, radius_r, center_atom)
                if env_bonds:
                    atom_map: dict = {}
                    for bond_idx in env_bonds:
                        bond = mol_3d.GetBondWithIdx(bond_idx)
                        atom_map[bond.GetBeginAtomIdx()] = -1
                        atom_map[bond.GetEndAtomIdx()] = -1
                    atom_map[center_atom] = -1
                    top_fragment = Chem.MolFragmentToSmiles(
                        mol_3d,
                        atomsToUse=list(atom_map.keys()),
                        bondsToUse=list(env_bonds),
                        canonical=True,
                    ) or None
            except Exception:
                top_fragment = None

    return {
        "smiles":        smiles,
        "sdf_block":     sdf_block,
        "atom_colors":   atom_colors,
        "n_atoms":       n_atoms,
        "n_toxic_atoms": n_toxic_atoms,
        "top_fragment":  top_fragment,
    }


if __name__ == "__main__":

    import uvicorn
    # Run server manually for testing: python src/api.py
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
