import React, { useMemo, useState } from 'react';
import axios from 'axios';
import Mol3DViewer from './Mol3DViewer';
import PipelineLog from './PipelineLog';
import { DecisionTrace } from './DecisionTrace';

const TARGET_COLS = [
  'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
  'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
];

const TOP_CANDIDATE_COUNT = 3;

function formatEndpointLabel(endpoint: string) { return endpoint; }

function formatProbability(value?: number | null) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return value.toFixed(2);
}

function formatPercentChange(originalValue?: number, candidateValue?: number) {
  if (typeof originalValue !== 'number' || typeof candidateValue !== 'number' || originalValue <= 0) return null;
  return Math.abs(((originalValue - candidateValue) / originalValue) * 100).toFixed(0);
}

function getRiskLabel(flaggedCount: number) {
  if (flaggedCount === 0) return { label: 'Clean', tone: 'text-emerald-300 border-emerald-400/30 bg-emerald-400/10' };
  if (flaggedCount <= 2) return { label: 'Moderate', tone: 'text-amber-300 border-amber-400/30 bg-amber-400/10' };
  return { label: 'High risk', tone: 'text-rose-300 border-rose-400/30 bg-rose-400/10' };
}

function getConfidenceLabel(oodSimilarity?: number) {
  if (typeof oodSimilarity !== 'number') return { label: 'Coverage unavailable', tone: 'text-slate-300 border-white/10 bg-white/5' };
  if (oodSimilarity >= 0.8) return { label: 'Known scaffold', tone: 'text-emerald-300 border-emerald-400/30 bg-emerald-400/10' };
  if (oodSimilarity >= 0.55) return { label: 'Analog scaffold', tone: 'text-amber-300 border-amber-400/30 bg-amber-400/10' };
  return { label: 'OOD warning', tone: 'text-rose-300 border-rose-400/30 bg-rose-400/10' };
}

function summarizeCandidate(candidate: any, originalProbabilities: Record<string, number>, flaggedEndpoints: string[]) {
  const endpointChanges = TARGET_COLS.map((endpoint) => {
    const originalValue = originalProbabilities?.[endpoint] ?? 0;
    const candidateValue = candidate?.toxicity_probs?.[endpoint] ?? 0;
    return { endpoint, originalValue, candidateValue, delta: originalValue - candidateValue, changed: candidateValue !== originalValue };
  });
  const flaggedChanges = endpointChanges.filter((c) => flaggedEndpoints.includes(c.endpoint)).sort((a, b) => b.delta - a.delta);
  const improvedOrStableCount = endpointChanges.filter((c) => c.delta >= 0).length;
  const worsenedCount = endpointChanges.filter((c) => c.delta < 0).length;
  return { endpointChanges, flaggedChanges, improvedOrStableCount, worsenedCount };
}

function calculateRadarPolygon(probabilities: Record<string, number> | null) {
  if (!probabilities) return '50,20 75,35 80,60 60,85 30,75 15,50 25,25';
  return TARGET_COLS.map((col, i) => {
    const angle = (i / 12) * Math.PI * 2 - Math.PI / 2;
    const r = (probabilities[col] || 0) * 40;
    return `${(50 + r * Math.cos(angle)).toFixed(1)},${(50 + r * Math.sin(angle)).toFixed(1)}`;
  }).join(' ');
}

export default function App() {
  const [smilesInput, setSmilesInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isTraceOpen, setIsTraceOpen] = useState(false);

  const analysis = useMemo(() => {
    const prediction = result?.prediction ?? {};
    const probabilities = prediction.probabilities ?? {};
    const flaggedEndpoints = TARGET_COLS
      .filter((col) => prediction.flags?.[col])
      .sort((l, r) => (probabilities[r] ?? 0) - (probabilities[l] ?? 0));
    const safeEndpoints = TARGET_COLS.filter((col) => !prediction.flags?.[col]);
    const topFragment = result?.primary_toxic_fragment || result?.shap_fragments?.find((i: any) => i.fragment)?.fragment || null;
    const bestCandidate = result?.pareto_candidates?.[0] ?? null;
    return { probabilities, flaggedEndpoints, safeEndpoints, topFragment, bestCandidate, risk: getRiskLabel(prediction.n_flagged ?? 0) };
  }, [result]);

  const handleAnalyze = async () => {
    if (!smilesInput) return;
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post('http://localhost:8000/analyze', { smiles: smilesInput }, { timeout: 120000 });
      setResult(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to analyze molecule');
    } finally {
      setLoading(false);
    }
  };

  const polygonPoints = calculateRadarPolygon(analysis.probabilities);
  const bestCandidateSummary = analysis.bestCandidate
    ? summarizeCandidate(analysis.bestCandidate, analysis.probabilities, analysis.flaggedEndpoints)
    : null;
  const candidateCards = (result?.pareto_candidates ?? []).slice(0, TOP_CANDIDATE_COUNT);

  return (
    <>
      {/* Background blobs */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-24 right-[-10%] h-[40rem] w-[40rem] rounded-full bg-primary/10 blur-[130px]" />
        <div className="absolute bottom-[-18%] left-[-12%] h-[32rem] w-[32rem] rounded-full bg-emerald-500/10 blur-[120px]" />
        <div className="absolute inset-0 opacity-[0.08] [background-image:linear-gradient(rgba(255,255,255,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.1)_1px,transparent_1px)] [background-size:56px_56px]" />
      </div>

      {/* Header */}
      <header className="fixed top-0 z-50 flex h-16 w-full items-center justify-between border-b border-white/5 bg-slate-950/80 px-4 backdrop-blur-2xl sm:px-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 text-primary">
            <span className="material-symbols-outlined text-[20px]">science</span>
          </div>
          <div>
            <div className="font-display text-base font-semibold tracking-tight text-slate-100">ToxNet Prescription Engine</div>
            <div className="text-[10px] uppercase tracking-[0.22em] text-slate-400">Chemist-facing toxicity report</div>
          </div>
        </div>
        <div className="flex flex-1 items-center justify-end gap-3 pl-4">
          <div className="flex flex-1 max-w-xl items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2">
            <span className="material-symbols-outlined text-[17px] text-primary">search</span>
            <input
              type="text"
              className="w-full border-none bg-transparent text-sm text-slate-100 outline-none placeholder:text-slate-500"
              placeholder="Enter SMILES, e.g. Nc1ccccc1"
              value={smilesInput}
              onChange={(e) => setSmilesInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
            />
          </div>
          <button
            onClick={() => setIsTraceOpen(true)}
            disabled={!result && !loading}
            className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-slate-800/80 px-4 py-2 text-sm font-semibold text-slate-300 transition-colors hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="material-symbols-outlined text-[16px]">plumbing</span>
            Trace Log
          </button>
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-full bg-emerald-400 px-4 py-2 text-sm font-semibold text-black transition-colors hover:bg-emerald-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="material-symbols-outlined text-[16px]">auto_fix_high</span>
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
      </header>

      {/* Bento Grid — fixed scroll container wraps the actual grid */}
      <main className="fixed inset-0 top-16 z-10 overflow-y-auto">
      <div className="bento-grid">

        {/* ── METRICS STRIP (full width row 1) ── */}
        <div style={{ gridArea: 'metrics' }} className="grid grid-cols-4 gap-2.5">
          <div className="glass col-span-1 flex flex-col justify-between rounded-2xl border border-white/10 bg-white/5 px-3 py-2.5 overflow-hidden">
            <div className="text-[9px] uppercase tracking-[0.3em] text-slate-400">Input SMILES</div>
            <div className="mt-1 truncate font-mono text-xs text-slate-100">{result?.input_smiles || 'Awaiting input…'}</div>
            {error && <div className="mt-1 truncate text-[9px] text-rose-300">{error}</div>}
          </div>
          <MetricCard label="Flagged" value={result ? `${result.prediction?.n_flagged ?? 0}/12` : '—'} hint={analysis.risk.label} tone={analysis.risk.tone} />
          <MetricCard label="SHAP Confidence" value={result?.alerts?.shap_confidence || '—'} hint="Structural evidence" tone="text-primary border-primary/20 bg-primary/10" />
          <MetricCard label="Candidates" value={result ? `${result.pareto_candidates?.length ?? 0}` : '—'} hint="Pareto-ranked" tone="text-amber-200 border-amber-400/20 bg-amber-400/10" />
        </div>

        {/* ── FLAGGED ENDPOINTS + SHAP (col A, rows 2) ── */}
        <section style={{ gridArea: 'flagged' }} className="glass flex flex-col rounded-2xl border border-white/10 bg-slate-950/40 p-4">
          <div className="flex shrink-0 items-center justify-between border-b border-white/10 pb-3 mb-3">
            <div>
              <p className="text-[9px] uppercase tracking-[0.3em] text-primary/80">Toxicity Profile</p>
              <h2 className="text-sm font-semibold text-slate-50">Flagged Endpoints</h2>
            </div>
            <span className="material-symbols-outlined text-[18px] text-rose-300">shield_alert</span>
          </div>
          <div className="flex-1 overflow-y-auto space-y-1.5 pr-0.5 min-h-0">
            {result && analysis.flaggedEndpoints.length > 0 ? analysis.flaggedEndpoints.map((ep) => (
              <div key={ep} className="flex items-center justify-between rounded-xl border border-rose-400/10 bg-rose-400/5 px-3 py-1.5">
                <span className="text-xs font-medium text-slate-100">{formatEndpointLabel(ep)}</span>
                <span className="font-mono text-xs text-rose-200">{formatProbability(analysis.probabilities?.[ep])}</span>
              </div>
            )) : (
              <div className="rounded-xl border border-emerald-400/10 bg-emerald-400/5 px-3 py-2 text-xs text-emerald-100/90">
                {result ? 'No endpoints exceed the calibrated threshold.' : 'Run analysis to populate.'}
              </div>
            )}
          </div>
          <div className="mt-3 shrink-0 border-t border-white/10 pt-3 space-y-1.5">
            <p className="text-[9px] uppercase tracking-[0.3em] text-slate-400 mb-2">SHAP Evidence</p>
            {result?.shap_fragments?.length ? result.shap_fragments.slice(0, 3).map((item: any, idx: number) => (
              <div key={`${item.bit}-${idx}`} className="flex items-center justify-between rounded-lg border border-white/10 bg-black/20 px-2.5 py-1.5">
                <span className="text-xs text-slate-200">Bit {item.bit} <span className="font-mono text-[10px] text-slate-400 ml-1 truncate">{item.fragment || '—'}</span></span>
                <span className="font-mono text-[10px] text-rose-200 shrink-0 ml-2">+{Number(item.importance ?? 0).toFixed(2)}</span>
              </div>
            )) : (
              <div className="rounded-lg border border-white/10 bg-black/20 px-2.5 py-1.5 text-xs text-slate-400">
                {result ? 'No SHAP fragments resolved.' : 'Awaiting analysis.'}
              </div>
            )}
          </div>
        </section>

        {/* ── 3D MOLECULE VIEWER (col B, row 2) ── */}
        <section style={{ gridArea: 'radar' }} className="glass flex flex-col rounded-2xl border border-primary/15 bg-slate-950/40 p-4">
          <div className="flex shrink-0 items-center justify-between border-b border-white/10 pb-3 mb-3">
            <div>
              <p className="text-[9px] uppercase tracking-[0.3em] text-primary/80">SHAP-Attributed</p>
              <h2 className="text-sm font-semibold text-slate-50">3D Structure Viewer</h2>
            </div>
            <div className="flex items-center gap-2">
              {result && (
                <span className="text-[9px] rounded-full border border-rose-400/20 bg-rose-400/10 px-2 py-0.5 text-rose-300 uppercase tracking-[0.2em]">
                  {result.prediction?.n_flagged ?? 0} flagged
                </span>
              )}
              <span className="material-symbols-outlined text-[18px] text-primary">deployed_code</span>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <Mol3DViewer
              smiles={result?.input_smiles ?? null}
              shapFragments={result?.shap_fragments}
              probabilities={analysis.probabilities}
              flaggedEndpoints={analysis.flaggedEndpoints}
            />
          </div>
        </section>

        {/* ── CANDIDATES (col C, spans rows 2+3) ── */}
        <section style={{ gridArea: 'cands' }} className="glass flex flex-col rounded-2xl border border-white/10 bg-slate-950/40 p-4">
          <div className="flex shrink-0 items-center justify-between border-b border-white/10 pb-3 mb-3">
            <div>
              <p className="text-[9px] uppercase tracking-[0.3em] text-primary/80">Suggested Modifications</p>
              <h2 className="text-sm font-semibold text-slate-50">Pareto-ranked Candidates</h2>
            </div>
            <span className="material-symbols-outlined text-[18px] text-amber-300">auto_fix_high</span>
          </div>
          <div className="flex-1 overflow-y-auto space-y-3 pr-0.5 min-h-0">
            {!result && !loading && (
              <div className="rounded-xl border border-white/10 bg-black/20 px-4 py-8 text-center text-sm text-slate-400">
                Run analysis to generate candidates.
              </div>
            )}
            {loading && [1, 2, 3].map(i => (
              <div key={i} className="animate-pulse rounded-xl border border-white/10 bg-white/5 p-3 space-y-2">
                <div className="h-2.5 w-20 rounded bg-white/10" />
                <div className="h-2 w-full rounded bg-white/10" />
                <div className="h-2 w-4/5 rounded bg-white/10" />
              </div>
            ))}
            {result && candidateCards.length === 0 && (
              <div className="rounded-xl border border-white/10 bg-black/20 px-4 py-6 text-sm text-slate-400">
                {result.pipeline_status === 'clean_no_action_needed'
                  ? 'Molecule is clean — no candidates needed.'
                  : 'No candidates passed the filters.'}
              </div>
            )}
            {candidateCards.map((candidate: any, index: number) => {
              const summary = summarizeCandidate(candidate, analysis.probabilities, analysis.flaggedEndpoints);
              const conf = getConfidenceLabel(candidate.ood_max_sim);
              const topChanges = summary.flaggedChanges.slice(0, 3);
              return (
                <article key={candidate.rank ?? index} className="rounded-xl border border-white/10 bg-white/5 p-3.5">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-black/20 px-2 py-0.5 text-[9px] uppercase tracking-[0.28em] text-slate-300">
                        Rank {candidate.rank ?? index + 1} · F{candidate.pareto_front ?? '-'}
                      </div>
                      <div className="mt-1.5 break-all font-mono text-xs text-slate-200">{candidate.smiles}</div>
                    </div>
                    <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[9px] uppercase tracking-[0.2em] ${conf.tone}`}>{conf.label}</span>
                  </div>
                  <div className="mt-2.5 space-y-1.5">
                    {topChanges.length ? topChanges.map(ch => {
                      const pct = formatPercentChange(ch.originalValue, ch.candidateValue);
                      return (
                        <div key={ch.endpoint} className="flex items-center justify-between rounded-lg border border-white/10 bg-black/20 px-2.5 py-1.5 text-xs">
                          <span className="text-slate-300">{formatEndpointLabel(ch.endpoint)}</span>
                          <span className="font-mono">
                            <span className="text-slate-400">{formatProbability(ch.originalValue)} → {formatProbability(ch.candidateValue)}</span>
                            <span className="ml-1.5 text-emerald-300">↓{pct}%</span>
                          </span>
                        </div>
                      );
                    }) : <div className="text-xs text-slate-500">No flagged endpoint changes.</div>}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5 text-[9px]">
                    <span className="rounded-full border border-white/10 bg-black/20 px-2 py-0.5 text-slate-400">SA {candidate.sa_score}</span>
                    <span className="rounded-full border border-white/10 bg-black/20 px-2 py-0.5 text-slate-400">SC {candidate.sc_score}</span>
                    <span className="rounded-full border border-white/10 bg-black/20 px-2 py-0.5 text-slate-400">Sim {(candidate.ood_max_sim ?? 0).toFixed(2)}</span>
                    <span className="rounded-full border border-white/10 bg-black/20 px-2 py-0.5 text-slate-400">{candidate.synth_verdict}</span>
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        {/* ── PRIMARY FRAGMENT (col A, row 3) ── */}
        <div style={{ gridArea: 'shap' }} className="glass flex flex-col rounded-2xl border border-white/10 bg-slate-950/40 p-4">
          <div className="flex shrink-0 items-center gap-2 mb-3">
            <span className="material-symbols-outlined text-[16px] text-primary">extension</span>
            <p className="text-[9px] uppercase tracking-[0.3em] text-primary/80">Primary Toxic Fragment</p>
          </div>
          <div className="flex-1 overflow-y-auto font-mono text-xs text-slate-100 break-all leading-relaxed min-h-0">
            {analysis.topFragment || 'No fragment resolved from the current molecule.'}
          </div>
          <div className="mt-3 shrink-0 space-y-1.5">
            <div className="flex justify-between rounded-lg border border-white/10 bg-black/20 px-3 py-1.5 text-xs">
              <span className="text-slate-400">Brenk filter</span>
              <span className={result?.alerts?.brenk_hit ? 'text-rose-300' : 'text-emerald-300'}>{result ? (result.alerts?.brenk_hit ? 'Matched' : 'Clear') : '—'}</span>
            </div>
            <div className="flex justify-between rounded-lg border border-white/10 bg-black/20 px-3 py-1.5 text-xs">
              <span className="text-slate-400">Alert count</span>
              <span className="text-slate-200">{result?.alerts?.alert_names?.length ?? (result ? 0 : '—')}</span>
            </div>
            <div className="flex justify-between rounded-lg border border-white/10 bg-black/20 px-3 py-1.5 text-xs">
              <span className="text-slate-400">Structural alerts</span>
              <span className="text-slate-300 truncate ml-2">{result?.alerts?.alert_names?.join(', ') || (result ? 'None' : '—')}</span>
            </div>
          </div>
        </div>

        {/* ── RECOMMENDATION (col B, row 3) ── */}
        <section style={{ gridArea: 'bottom' }} className="glass flex flex-col rounded-2xl border border-emerald-400/15 bg-emerald-400/5 p-4">
          <div className="flex shrink-0 items-center justify-between border-b border-emerald-400/10 pb-3 mb-3">
            <div>
              <p className="text-[9px] uppercase tracking-[0.3em] text-emerald-300/80">Recommendation</p>
              <h2 className="text-sm font-semibold text-slate-50">Wet-lab Decision</h2>
            </div>
            <span className="material-symbols-outlined text-[18px] text-emerald-300">verified</span>
          </div>
          <div className="flex-1 overflow-y-auto text-xs text-slate-300 leading-relaxed min-h-0">
            {bestCandidateSummary ? (
              <>
                <p className="text-slate-100">
                  Rank {analysis.bestCandidate.rank ?? 1} is the preferred option — Pareto-first, lowers active toxic endpoints, synthesis-friendly.
                </p>
                <p className="mt-2 text-emerald-200">
                  {bestCandidateSummary.flaggedChanges.length
                    ? `Improves ${bestCandidateSummary.flaggedChanges.length} flagged endpoint(s).`
                    : 'Strongest candidate after all filters.'
                  }{' '}{bestCandidateSummary.improvedOrStableCount}/12 endpoints unchanged or better.
                </p>
                <p className="mt-2 text-slate-400">Wet-lab validation required before clinical use.</p>
              </>
            ) : (
              <p>{result ? 'No ranked candidate available.' : 'Run analysis to generate a recommendation.'}</p>
            )}
          </div>
        </section>

        {/* ══════════════════════════════════════════════════════════════════
            ROW 4 — THREE GROUNDBREAKING FRAMEWORKS
            ══════════════════════════════════════════════════════════════════ */}

        {/* ── TOXICITY CONSTELLATION (col A, row 4) ── */}
        <section style={{ gridArea: 'constell' }} className="glass constellation-glow flex min-h-0 flex-col rounded-2xl border border-violet-400/20 bg-violet-500/5 p-4">
          <div className="flex shrink-0 items-center justify-between border-b border-violet-400/10 pb-3 mb-3">
            <div>
              <p className="text-[9px] uppercase tracking-[0.3em] text-violet-300/80">Novel Framework</p>
              <h2 className="text-sm font-semibold text-slate-50">Toxicity Constellation</h2>
            </div>
            <span className="material-symbols-outlined text-[18px] text-violet-300">scatter_plot</span>
          </div>
          <div className="flex-1 overflow-y-auto min-h-0 space-y-2">
            {!result && <p className="text-xs text-slate-500">Run analysis to classify this molecule into its mechanistic archetype.</p>}
            {result && !result.constellation && <p className="text-xs text-slate-500">Constellation unavailable — run precompute_frameworks.py.</p>}
            {result?.constellation && result.constellation.constellation_name !== 'UNAVAILABLE' && (() => {
              const c = result.constellation;
              return (
                <>
                  <div className="flex items-center gap-2">
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-violet-400/30 bg-violet-400/15 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-violet-200">
                      <span className="material-symbols-outlined text-[12px]">auto_awesome</span>
                      {c.constellation_name}
                    </span>
                    <span className={`text-[9px] rounded-full border px-2 py-0.5 uppercase tracking-[0.2em] ${c.proximity_label === 'CANONICAL' ? 'text-violet-200 border-violet-400/30 bg-violet-400/10' : c.proximity_label === 'TYPICAL' ? 'text-amber-200 border-amber-400/30 bg-amber-400/10' : 'text-slate-400 border-white/10 bg-white/5'}`}>
                      {c.proximity_label}
                    </span>
                  </div>
                  <div className="rounded-xl border border-violet-400/10 bg-black/20 px-3 py-2">
                    <p className="text-[10px] uppercase tracking-[0.2em] text-violet-300/60 mb-1">Mechanism</p>
                    <p className="text-xs text-slate-200 leading-relaxed">{c.mechanism_hint}</p>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
                    <p className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-1.5">Dominant Endpoints</p>
                    <div className="flex flex-wrap gap-1">
                      {(c.dominant_endpoints || []).map((ep: string) => (
                        <span key={ep} className="rounded-full border border-violet-400/20 bg-violet-400/10 px-2 py-0.5 text-[9px] text-violet-200">{ep}</span>
                      ))}
                    </div>
                  </div>
                  <div className="flex justify-between text-[10px]">
                    <span className="text-slate-400">Archetype distance</span>
                    <span className="font-mono text-slate-200">{(c.distance ?? 0).toFixed(3)}</span>
                  </div>
                </>
              );
            })()}
          </div>
        </section>

        {/* ── SCAFFOLD BRITTLENESS / STBI (col B, row 4) ── */}
        <section style={{ gridArea: 'stbi' }} className="glass flex min-h-0 flex-col rounded-2xl border border-orange-400/15 bg-orange-500/5 p-4">
          <div className="flex shrink-0 items-center justify-between border-b border-orange-400/10 pb-3 mb-3">
            <div>
              <p className="text-[9px] uppercase tracking-[0.3em] text-orange-300/80">Novel Framework</p>
              <h2 className="text-sm font-semibold text-slate-50">Scaffold Brittleness</h2>
            </div>
            <span className="material-symbols-outlined text-[18px] text-orange-300">crisis_alert</span>
          </div>
          <div className="flex-1 overflow-y-auto min-h-0 space-y-2">
            {/* No result yet */}
            {!result && <p className="text-xs text-slate-500">Run analysis to compute scaffold toxicity brittleness index.</p>}

            {result?.stbi && (() => {
              const s = result.stbi;
              const hasScore = typeof s.stbi === 'number';
              const stbiVal  = hasScore ? s.stbi : 0;
              const pct      = Math.round(stbiVal * 100);

              const assessTone =
                s.assessment === 'EXTREME'     ? 'text-rose-300 border-rose-400/30 bg-rose-400/10'    :
                s.assessment === 'MODERATE'    ? 'text-amber-300 border-amber-400/30 bg-amber-400/10' :
                s.assessment === 'ROBUST'      ? 'text-emerald-300 border-emerald-400/30 bg-emerald-400/10' :
                s.assessment === 'UNSEEN'      ? 'text-slate-400 border-white/10 bg-white/5'          :
                s.assessment === 'UNAVAILABLE' ? 'text-slate-500 border-white/5 bg-white/3'           :
                'text-slate-400 border-white/10 bg-white/5';

              return (
                <>
                  {/* Assessment badge + live tag */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] ${assessTone}`}>
                      {s.assessment}
                    </span>
                    {s.live_computed && (
                      <span className="rounded-full border border-orange-400/30 bg-orange-400/10 px-2 py-0.5 text-[9px] text-orange-300 uppercase tracking-[0.2em]">
                        live computed
                      </span>
                    )}
                    {hasScore && (
                      <span className="ml-auto font-mono text-sm font-bold text-slate-100">
                        {stbiVal.toFixed(2)}
                        <span className="text-[10px] text-slate-400 ml-1">/ 1.0</span>
                      </span>
                    )}
                  </div>

                  {/* Score gauge bar — only when we have a real number */}
                  {hasScore && (
                    <div className="relative h-2 w-full overflow-hidden rounded-full bg-white/10">
                      <div
                        className="stbi-gauge absolute inset-0"
                        style={{ clipPath: `inset(0 ${100 - pct}% 0 0)` }}
                      />
                      <div
                        className="absolute top-0 h-full w-0.5 bg-white/80"
                        style={{ left: `${pct}%` }}
                      />
                    </div>
                  )}

                  {/* UNSEEN state — no score but explain why */}
                  {!hasScore && s.assessment === 'UNSEEN' && (
                    <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2 flex items-start gap-2">
                      <span className="material-symbols-outlined text-[14px] text-slate-400 shrink-0 mt-0.5">info</span>
                      <p className="text-[10px] text-slate-400 leading-relaxed">
                        Scaffold not present in the Tox21 training set with both toxic and safe members.
                        STBI requires cross-class scaffold members to measure brittleness.
                        This molecule is outside the applicability domain for brittleness scoring.
                      </p>
                    </div>
                  )}

                  {/* Message box */}
                  <div className="rounded-xl border border-orange-400/10 bg-black/20 px-3 py-2">
                    <p className="text-[10px] leading-relaxed text-slate-300">{s.message}</p>
                  </div>

                  {/* Scaffold SMILES */}
                  {s.scaffold && (
                    <div className="rounded-lg border border-white/10 bg-black/20 px-2.5 py-1.5">
                      <p className="text-[9px] uppercase tracking-[0.2em] text-slate-400 mb-0.5">Murcko Scaffold</p>
                      <p className="font-mono text-[10px] text-slate-300 break-all">{s.scaffold}</p>
                    </div>
                  )}

                  {/* Per-endpoint scores if available */}
                  {hasScore && s.endpoint_scores && Object.keys(s.endpoint_scores).length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[9px] uppercase tracking-[0.2em] text-slate-400">Per-Endpoint STBI</p>
                      {Object.entries(s.endpoint_scores as Record<string, number>)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 5)
                        .map(([ep, score]) => (
                          <div key={ep} className="flex items-center gap-2">
                            <span className="text-[10px] text-slate-400 w-24 shrink-0">{ep}</span>
                            <div className="flex-1 h-1 rounded-full bg-white/10 overflow-hidden">
                              <div
                                className="h-full rounded-full"
                                style={{
                                  width: `${Math.round(score * 100)}%`,
                                  background: score >= 0.8 ? '#f87171' : score >= 0.55 ? '#fb923c' : '#34d399',
                                }}
                              />
                            </div>
                            <span className="font-mono text-[10px] text-slate-300 w-8 text-right">{score.toFixed(2)}</span>
                          </div>
                        ))}
                    </div>
                  )}
                </>
              );
            })()}
          </div>
        </section>

        {/* ── ESCAPE PATH / MTEP (col C, row 4) ── */}
        <section style={{ gridArea: 'mtep' }} className="glass flex min-h-0 flex-col rounded-2xl border border-cyan-400/15 bg-cyan-500/5 p-4">
          <div className="flex shrink-0 items-center justify-between border-b border-cyan-400/10 pb-3 mb-3">
            <div>
              <p className="text-[9px] uppercase tracking-[0.3em] text-cyan-300/80">Novel Framework</p>
              <h2 className="text-sm font-semibold text-slate-50">Escape Path Analysis</h2>
            </div>
            <span className="material-symbols-outlined text-[18px] text-cyan-300">explore</span>
          </div>
          <div className="flex-1 overflow-y-auto min-h-0 space-y-2">
            {!result && <p className="text-xs text-slate-500">Run analysis to compute gradient-based escape pressures per endpoint.</p>}
            {result?.escape_path && (() => {
              const ep = result.escape_path;
              const summary = ep.summary;
              const perEndpoint = ep.per_endpoint ?? {};
              const flaggedEps = Object.entries(perEndpoint).filter(([, v]: [string, any]) => v.flagged);

              const prognosisTone = summary?.overall_prognosis === 'OPTIMISTIC'
                ? 'text-emerald-300 border-emerald-400/30 bg-emerald-400/10'
                : summary?.overall_prognosis === 'CHALLENGING'
                ? 'text-amber-300 border-amber-400/30 bg-amber-400/10'
                : summary?.overall_prognosis === 'REDESIGN_NEEDED'
                ? 'text-rose-300 border-rose-400/30 bg-rose-400/10'
                : 'text-slate-400 border-white/10 bg-white/5';

              const diffBadge = (d: string) => {
                if (d === 'EASY')    return 'badge-easy';
                if (d === 'HARD')    return 'badge-hard';
                if (d === 'TRAPPED') return 'badge-trapped';
                return 'badge-clean';
              };

              return (
                <>
                  {summary && (
                    <div className="flex items-center gap-2">
                      <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.15em] ${prognosisTone}`}>
                        {summary.overall_prognosis?.replace('_', ' ')}
                      </span>
                      {summary.flagged_count > 0 && (
                        <span className="text-[10px] text-slate-400">
                          {summary.easy_count}E · {summary.hard_count}H · {summary.trapped_count}T
                        </span>
                      )}
                    </div>
                  )}
                  {summary?.prognosis_message && (
                    <div className="rounded-xl border border-cyan-400/10 bg-black/20 px-3 py-2">
                      <p className="text-[10px] leading-relaxed text-slate-300">{summary.prognosis_message}</p>
                    </div>
                  )}
                  {flaggedEps.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-[9px] uppercase tracking-[0.2em] text-slate-400">Per-Endpoint Escape Difficulty</p>
                      {flaggedEps.map(([epName, v]: [string, any]) => (
                        <div key={epName} className="flex items-center justify-between rounded-lg border border-white/10 bg-black/20 px-2.5 py-1.5">
                          <span className="text-xs text-slate-300">{epName}</span>
                          <div className="flex items-center gap-2">
                            {v.gradient_norm !== null && (
                              <span className="font-mono text-[10px] text-slate-500">∇={v.gradient_norm?.toFixed(4)}</span>
                            )}
                            <span className={`rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.15em] ${diffBadge(v.difficulty)}`}>
                              {v.difficulty}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  {flaggedEps.length === 0 && result && (
                    <p className="text-xs text-emerald-300">No endpoints flagged — molecule is clean.</p>
                  )}
                </>
              );
            })()}
          </div>
        </section>
        {/* ── PIPELINE ACTIVITY LOG (full-width bottom row) ── */}
        <PipelineLog isLoading={loading} />

      </div>
      </main>

      {/* Decision Trace Overlay */}
      <DecisionTrace isOpen={isTraceOpen} onClose={() => setIsTraceOpen(false)} />

    </>
  );
}


function MetricCard({
  label, value, hint, tone,
}: {
  label: string;
  value: React.ReactNode;
  hint: string;
  tone: string;
}) {
  return (
    <div className={`rounded-2xl border px-3 py-2.5 ${tone}`}>
      <div className="text-[9px] uppercase tracking-[0.28em] text-current/80">{label}</div>
      <div className="mt-1.5 font-display text-xl font-semibold tracking-tight text-slate-50">{value}</div>
      <div className="mt-0.5 text-[10px] text-current/80">{hint}</div>
    </div>
  );
}
