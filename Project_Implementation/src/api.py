import os
import sys
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure src is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.train import get_device
from src.explain import load_pipeline_from_artifact
from src.bioisostere import load_chembl_cache
from src.prescription_pipeline import run_prescription_pipeline

# ─── Global State ─────────────────────────────────────────────────────────────

# We load ML models and caches ONCE at application startup.
ml_state = {}

TARGET_COLS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

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
    
    print("✅ API is ready to receive requests.")
    yield
    
    print("🛑 Shutting down API. Cleaning up ML resources...")
    ml_state.clear()

# ─── FastAPI App Initialization ───────────────────────────────────────────────

app = FastAPI(
    title="ToxNet Prescription Engine",
    description="Structural Optimization and Toxicity Prediction API",
    version="1.0.0",
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

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "online", "message": "ToxNet Prescription Engine API is running."}

@app.post("/analyze")
def analyze_molecule(req: AnalysisRequest):
    """
    Run the full 4-step prescription pipeline on a single SMILES string.
    """
    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES string cannot be empty.")
    
    try:
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
        
        if result['pipeline_status'] == 'error:invalid_smiles':
            raise HTTPException(status_code=400, detail="Invalid SMILES string provided.")
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run server manually for testing: python src/api.py
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
