import React, { useState } from 'react';
import axios from 'axios';

// The 12 endpoints
const TARGET_COLS = [
  'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
  'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
];

function calculateRadarPolygon(probabilities: Record<string, number> | null) {
  if (!probabilities) {
    // Default placeholder polygon
    return "50,20 75,35 80,60 60,85 30,75 15,50 25,25";
  }
  
  // Map 12 points around a circle. Center is 50,50. Max radius is 40.
  const points = TARGET_COLS.map((col, i) => {
    const angle = (i / 12) * Math.PI * 2 - Math.PI / 2; // Start at top
    const prob = probabilities[col] || 0;
    // Map prob (0 to 1) to radius (0 to 40)
    const r = prob * 40;
    const x = 50 + r * Math.cos(angle);
    const y = 50 + r * Math.sin(angle);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  
  return points.join(' ');
}

export default function App() {
  const [smilesInput, setSmilesInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!smilesInput) return;
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post('http://localhost:8000/analyze', {
        smiles: smilesInput
      }, { timeout: 120000 }); // 2 min — live ChEMBL queries can take ~60s
      setResult(res.data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || "Failed to analyze molecule");
    } finally {
      setLoading(false);
    }
  };

  const polygonPoints = calculateRadarPolygon(result?.prediction?.probabilities);

  return (
    <>
      {/* Atmospheric Deep Glows */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] right-[-5%] w-[800px] h-[800px] rounded-full bg-primary/5 blur-[120px]"></div>
        <div className="absolute bottom-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-emerald-500/5 blur-[100px]"></div>
      </div>

      {/* TopAppBar */}
      <header className="fixed top-0 w-full h-12 z-50 flex items-center justify-between px-6 bg-slate-950/80 backdrop-blur-xl text-emerald-500 font-inter antialiased tracking-tight text-xs border-b border-white/5 shadow-none transition-all duration-200 ease-in-out">
        <div className="flex items-center gap-6">
          <div className="text-lg font-black tracking-tighter text-slate-100 uppercase">ToxNet</div>
          <div className="flex items-center gap-2 px-3 py-1 bg-surface-container-high rounded-full border border-white/5 text-slate-400 w-96">
            <span className="material-symbols-outlined text-[16px]">science</span>
            <input 
              type="text" 
              className="bg-transparent border-none outline-none w-full text-slate-100" 
              placeholder="Enter SMILES string (e.g. Nc1ccccc1)" 
              value={smilesInput}
              onChange={(e) => setSmilesInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
            />
          </div>
          <button 
            onClick={handleAnalyze}
            disabled={loading}
            className="bg-emerald-500 text-black px-4 py-1 rounded font-bold tracking-wider uppercase disabled:opacity-50 hover:bg-emerald-400 transition-colors"
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
        <div className="flex items-center gap-4">
          <span className="material-symbols-outlined hover:bg-white/5 hover:text-emerald-400 p-1 rounded transition-colors cursor-pointer">notifications</span>
          <span className="material-symbols-outlined hover:bg-white/5 hover:text-emerald-400 p-1 rounded transition-colors cursor-pointer">account_circle</span>
        </div>
      </header>

      {/* SideNavBar */}
      <nav className="fixed left-0 top-12 h-[calc(100vh-48px)] w-64 z-40 flex flex-col pt-8 bg-slate-950/90 backdrop-blur-2xl text-emerald-500 font-inter text-sm font-medium tracking-wide border-r border-white/5 shadow-[4px_0_24px_rgba(0,0,0,0.5)]">
        <div className="px-6 mb-8">
          <div className="text-emerald-500 font-bold tracking-widest text-[11px] mb-1">TOXNET ENGINE</div>
          <div className="text-slate-500 text-[10px] tracking-widest uppercase">V2.0.4-PRE</div>
        </div>
        <div className="flex flex-col flex-1">
          <a className="flex items-center gap-3 bg-emerald-500/10 text-emerald-400 border-l-2 border-emerald-500 px-4 py-3 hover:bg-emerald-500/5 transition-colors" href="#">
            <span className="material-symbols-outlined">dashboard</span>
            <span>Dashboard</span>
          </a>
          <a className="flex items-center gap-3 text-slate-500 px-4 py-3 hover:text-slate-200 hover:bg-emerald-500/5 transition-colors" href="#">
            <span className="material-symbols-outlined">query_stats</span>
            <span>Molecular Analysis</span>
          </a>
          <a className="flex items-center gap-3 text-slate-500 px-4 py-3 hover:text-slate-200 hover:bg-emerald-500/5 transition-colors" href="#">
            <span className="material-symbols-outlined">biotech</span>
            <span>Clinical Trials</span>
          </a>
        </div>
      </nav>

      {/* Main Content Canvas */}
      <main className="ml-64 mt-12 h-[calc(100vh-48px)] p-container-padding pb-[60px] overflow-y-auto relative z-10">
        
        {error && (
          <div className="bg-error-container/20 border border-error/30 rounded-lg p-stack-sm mb-gutter flex items-center justify-between shadow-[0_0_20px_rgba(255,180,171,0.05)] backdrop-blur-md">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-error/10 flex items-center justify-center border border-error/20">
                <span className="material-symbols-outlined text-error text-[16px]">error</span>
              </div>
              <div>
                <h3 className="font-body-sm text-body-sm text-error font-semibold">Error</h3>
                <p className="font-body-sm text-body-sm text-error/80 text-xs">{error}</p>
              </div>
            </div>
          </div>
        )}

        {result && result.prediction.n_flagged > 0 && (
          <div className="bg-error-container/20 border border-error/30 rounded-lg p-stack-sm mb-gutter flex items-center justify-between shadow-[0_0_20px_rgba(255,180,171,0.05)] backdrop-blur-md">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-error/10 flex items-center justify-center border border-error/20">
                <span className="material-symbols-outlined text-error text-[16px]">warning</span>
              </div>
              <div>
                <h3 className="font-body-sm text-body-sm text-error font-semibold">Toxicity Alert: {result.prediction.n_flagged} endpoints flagged</h3>
                <p className="font-body-sm text-body-sm text-error/80 text-xs">
                  {result.alerts.brenk_hit ? `Structural Alert found: ${result.alerts.alert_names.join(', ')}.` : 'No known structural alerts.'} 
                  SHAP Confidence: {result.alerts.shap_confidence}
                </p>
              </div>
            </div>
            <button className="px-3 py-1.5 border border-error/30 text-error rounded font-label-caps text-label-caps hover:bg-error/10 transition-colors">Review Logs</button>
          </div>
        )}

        {result && result.prediction.n_flagged === 0 && (
           <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-stack-sm mb-gutter flex items-center justify-between shadow-[0_0_20px_rgba(16,185,129,0.05)] backdrop-blur-md">
           <div className="flex items-center gap-3">
             <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center border border-emerald-500/30">
               <span className="material-symbols-outlined text-emerald-400 text-[16px]">check_circle</span>
             </div>
             <div>
               <h3 className="font-body-sm text-body-sm text-emerald-400 font-semibold">Molecule is Safe</h3>
               <p className="font-body-sm text-body-sm text-emerald-400/80 text-xs">No toxicity endpoints flagged.</p>
             </div>
           </div>
         </div>
        )}

        {/* 3-Column Intelligent Grid */}
        <div className="grid grid-cols-12 gap-gutter h-full">
          
          {/* Left Column: Structural Intelligence */}
          <div className="col-span-3 flex flex-col gap-gutter h-full">
            <div className="bg-surface-container/60 border border-outline-variant/30 rounded-xl p-stack-md flex flex-col h-full shadow-lg backdrop-blur-xl">
              <div className="flex items-center justify-between mb-stack-md border-b border-white/5 pb-2">
                <h2 className="font-label-caps text-label-caps text-primary uppercase tracking-widest">Structural Intelligence</h2>
                <span className="material-symbols-outlined text-outline text-[16px]">view_in_ar</span>
              </div>
              
              {/* High-fidelity molecule viewer */}
              <div className="bg-white rounded-lg p-4 mb-stack-md relative h-48 flex items-center justify-center shadow-inner border border-white/10 overflow-hidden group">
                <div className="absolute inset-0 bg-gradient-to-tr from-transparent to-black/5 mix-blend-multiply"></div>
                
                {result ? (
                  <img 
                    src={`https://cactus.nci.nih.gov/chemical/structure/${encodeURIComponent(result.input_smiles)}/image`}
                    alt="Molecule 2D Structure"
                    className="w-full h-full object-contain mix-blend-darken filter contrast-125"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                      (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
                    }}
                  />
                ) : (
                  <div className="text-slate-400 font-mono text-sm uppercase tracking-widest">
                    Awaiting Input...
                  </div>
                )}
                <div className="hidden text-slate-900 font-mono font-bold text-center break-all w-full absolute px-4">
                  {result?.input_smiles}
                </div>
              </div>
              
              {/* Alerts & Confidence */}
              <div className="flex-1 flex flex-col gap-stack-sm overflow-y-auto pr-2">
                <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase mb-1">Risk Explainability (SHAP)</h3>
                
                {result ? result.shap_top_bits.map((bitInfo: any, idx: number) => (
                  <div key={idx} className="flex flex-col gap-2 mb-2">
                    <div className="flex items-center justify-between">
                      <span className="font-mono-data text-mono-data text-on-surface text-[11px]">Bit {bitInfo[0]}</span>
                      <span className="font-mono-data text-mono-data text-error font-bold text-[11px]">+{bitInfo[1].toFixed(2)}</span>
                    </div>
                    <div className="h-1 w-full bg-surface-bright rounded-full overflow-hidden">
                      <div className="h-full bg-error rounded-full shadow-[0_0_8px_rgba(255,180,171,0.8)]" style={{ width: `${Math.min((bitInfo[1]*100)+20, 100)}%` }}></div>
                    </div>
                  </div>
                )) : (
                  <div className="text-xs text-slate-500 italic">Run analysis to see SHAP values</div>
                )}
                
              </div>
            </div>
          </div>

          {/* Center Column: Toxicity Profile */}
          <div className="col-span-4 flex flex-col gap-gutter h-full">
            <div className="bg-surface-container-low/80 border border-outline-variant/30 rounded-xl p-stack-md flex flex-col h-full shadow-lg backdrop-blur-xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-[80px] pointer-events-none"></div>
              
              <div className="flex items-center justify-between mb-stack-lg border-b border-white/5 pb-2 z-10">
                <h2 className="font-label-caps text-label-caps text-primary uppercase tracking-widest">Toxicity Profile (12-Endpoint)</h2>
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${result ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]' : 'bg-slate-500'}`}></span>
                  <span className="font-mono-data text-[10px] text-on-surface-variant uppercase tracking-wider">{result ? 'Live' : 'Standby'}</span>
                </div>
              </div>
              
              {/* Radar Chart Placeholder (CSS Art) */}
              <div className="flex-1 flex items-center justify-center relative z-10 min-h-[250px]">
                <div className="relative w-[240px] h-[240px]">
                  <div className="absolute inset-0 border border-white/10 rounded-full"></div>
                  <div className="absolute inset-[30px] border border-white/10 rounded-full"></div>
                  <div className="absolute inset-[60px] border border-white/10 rounded-full"></div>
                  <div className="absolute inset-[90px] border border-white/20 rounded-full bg-surface-bright/20 backdrop-blur-sm"></div>
                  
                  <div className="absolute top-0 bottom-0 left-1/2 w-[1px] bg-white/10 -translate-x-1/2"></div>
                  <div className="absolute left-0 right-0 top-1/2 h-[1px] bg-white/10 -translate-y-1/2"></div>
                  <div className="absolute top-0 bottom-0 left-1/2 w-[1px] bg-white/10 -translate-x-1/2 rotate-45"></div>
                  <div className="absolute top-0 bottom-0 left-1/2 w-[1px] bg-white/10 -translate-x-1/2 -rotate-45"></div>
                  
                  <svg className="absolute inset-0 w-full h-full drop-shadow-[0_0_15px_rgba(255,180,171,0.2)]" viewBox="0 0 100 100">
                    <polygon className="opacity-80 transition-all duration-1000" fill="rgba(255, 180, 171, 0.15)" points={polygonPoints} stroke="#ffb4ab" strokeWidth="0.5"></polygon>
                  </svg>

                  {/* Render points if result exists */}
                  {result && TARGET_COLS.map((col, i) => {
                     const angle = (i / 12) * Math.PI * 2 - Math.PI / 2;
                     const prob = result.prediction.probabilities[col] || 0;
                     const r = prob * 40;
                     const x = 50 + r * Math.cos(angle);
                     const y = 50 + r * Math.sin(angle);
                     const isFlagged = result.prediction.flags[col];
                     if(prob < 0.1) return null; // Only show significant dots
                     return (
                       <div key={col} className={`absolute w-1.5 h-1.5 rounded-full -translate-x-1/2 -translate-y-1/2 ${isFlagged ? 'bg-error shadow-[0_0_10px_rgba(255,180,171,1)]' : 'bg-tertiary shadow-[0_0_10px_rgba(231,195,101,1)]'}`} style={{left: `${x}%`, top: `${y}%`}}></div>
                     )
                  })}
                </div>
              </div>
              
              {/* Uncertainty Panel */}
              <div className="mt-auto bg-surface-container border border-white/5 rounded-lg p-3 z-10">
                <h3 className="font-label-caps text-[9px] text-on-surface-variant uppercase tracking-widest mb-2 border-b border-white/5 pb-1">Prediction Stats</h3>
                <div className="flex flex-col gap-1">
                  <div className="flex justify-between items-center py-1 border-b border-white/5">
                    <span className="font-body-sm text-[11px] text-on-surface">Mean Probability</span>
                    <span className="font-mono-data text-[11px] text-primary">{result ? result.prediction.mean_prob : 'N/A'}</span>
                  </div>
                  <div className="flex justify-between items-center py-1">
                    <span className="font-body-sm text-[11px] text-on-surface">Flagged Endpoints</span>
                    <span className="font-mono-data text-[11px] text-error">{result ? `${result.prediction.n_flagged} / 12` : 'N/A'}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Prescription Engine */}
          <div className="col-span-5 flex flex-col gap-gutter h-full">
            <div className="bg-surface-container-high/50 border border-outline-variant/30 rounded-xl flex flex-col h-full shadow-lg backdrop-blur-md overflow-hidden">
              <div className="p-stack-md pb-2 border-b border-white/5 bg-surface-container-highest/30">
                <div className="flex items-center justify-between mb-1">
                  <h2 className="font-label-caps text-label-caps text-primary uppercase tracking-widest">Prescription Engine</h2>
                  <span className="material-symbols-outlined text-primary text-[14px]">auto_fix_high</span>
                </div>
                <p className="font-body-sm text-[11px] text-on-surface-variant">Pareto-Ranked Bioisostere Candidates</p>
              </div>
              
              <div className="flex-1 overflow-y-auto">
                <table className="w-full text-left border-collapse">
                  <thead className="sticky top-0 bg-surface-container-highest/90 backdrop-blur-md z-10">
                    <tr>
                      <th className="py-2 px-3 font-label-caps text-[9px] text-on-surface-variant uppercase tracking-wider border-b border-white/10 w-8 text-center">#</th>
                      <th className="py-2 px-3 font-label-caps text-[9px] text-on-surface-variant uppercase tracking-wider border-b border-white/10">Candidate (SMILES)</th>
                      <th className="py-2 px-3 font-label-caps text-[9px] text-on-surface-variant uppercase tracking-wider border-b border-white/10 text-right">Tox Δ</th>
                      <th className="py-2 px-3 font-label-caps text-[9px] text-on-surface-variant uppercase tracking-wider border-b border-white/10 text-right">Synthesizability</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono-data text-[11px] text-on-surface">
                    
                    {!result && !loading && (
                      <tr><td colSpan={4} className="text-center py-8 text-slate-500">Run analysis to generate candidates</td></tr>
                    )}

                    {/* Skeleton loader while waiting for live ChEMBL results */}
                    {loading && (
                      [1,2,3].map(i => (
                        <tr key={i} className="animate-pulse">
                          <td className="py-3 px-3 border-b border-white/5"><div className="h-3 w-4 bg-white/10 rounded mx-auto"></div></td>
                          <td className="py-3 px-3 border-b border-white/5"><div className="h-3 bg-white/10 rounded w-3/4"></div></td>
                          <td className="py-3 px-3 border-b border-white/5"><div className="h-3 bg-white/10 rounded w-8 ml-auto"></div></td>
                          <td className="py-3 px-3 border-b border-white/5"><div className="h-3 bg-white/10 rounded w-16 ml-auto"></div></td>
                        </tr>
                      ))
                    )}

                    {result && result.pareto_candidates?.length === 0 && (
                      <tr><td colSpan={4} className="text-center py-8 text-slate-500 italic">
                        {result.pipeline_status === 'clean_no_action_needed'
                          ? '✅ Molecule is clean — no bioisostere replacement needed.'
                          : 'No viable candidates found. Try a known toxin like BPA: CC(C)(c1ccc(O)cc1)c1ccc(O)cc1'}
                      </td></tr>
                    )}

                    {result && result.pareto_candidates?.map((cand: any, idx: number) => {
                      const toxDelta = result.prediction.n_flagged - cand.n_flagged;
                      const toxDeltaClass =
                        toxDelta > 0 ? 'text-emerald-400' : toxDelta < 0 ? 'text-error' : 'text-on-surface-variant';
                      const toxDeltaLabel = `${toxDelta > 0 ? '+' : ''}${toxDelta}`;

                      return (
                      <tr key={idx} className="group hover:bg-white/5 transition-colors cursor-pointer relative">
                        <td className="py-3 px-3 text-center border-b border-white/5 text-primary">{cand.rank ?? idx + 1}</td>
                        <td className="py-3 px-3 border-b border-white/5 truncate max-w-[120px]" title={cand.smiles}>
                          {cand.smiles}
                        </td>
                        <td className={`py-3 px-3 border-b border-white/5 text-right font-bold ${toxDeltaClass}`}>
                          {toxDeltaLabel}
                        </td>
                        <td className="py-3 px-3 border-b border-white/5 text-right text-on-surface-variant">
                          F{cand.pareto_front ?? '-'} · SC: {cand.sc_score}
                        </td>
                        <td className="absolute left-0 top-0 bottom-0 w-[2px] bg-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity"></td>
                      </tr>
                    )})}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
