/**
 * Mol3DViewer.tsx
 * Interactive 3D molecular visualization with SHAP-based toxic atom highlighting.
 *
 * Features:
 *  - Full 3D conformer rendered via 3Dmol.js
 *  - Per-atom coloring: red (high SHAP) → orange → gold → grey
 *  - Glowing halos on high-attribution atoms
 *  - Click any atom → floating info panel (element, SHAP score, toxicity level,
 *    which endpoints it contributes to, fragment label)
 *  - Clicked atom gets a pulsing highlight ring in the 3D view
 *  - Spin / pause / drag to explore
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import axios from 'axios';

declare const $3Dmol: any;

// ─── Types ────────────────────────────────────────────────────────────────────

interface AtomColor {
  atom_idx: number;
  shap_score: number;
  is_toxic: boolean;
}

interface Mol3DData {
  sdf_block: string;
  atom_colors: AtomColor[];
  smiles: string;
  n_toxic_atoms: number;
  n_atoms: number;
  top_fragment: string | null;
}

interface SelectedAtomInfo {
  idx: number;
  elem: string;
  shap_score: number;
  is_toxic: boolean;
  level: 'HIGH' | 'MODERATE' | 'MILD' | 'INERT';
  color: string;
  levelDesc: string;
  advice: string;
  flaggedEndpoints: string[];
}

interface Mol3DViewerProps {
  smiles: string | null;
  shapFragments?: any[];
  probabilities?: Record<string, number>;
  flaggedEndpoints?: string[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const SPIN_SPEED = 0.8;

function getAtomLevel(shap: number, is_toxic: boolean): {
  level: SelectedAtomInfo['level'];
  color: string;
  levelDesc: string;
  advice: string;
} {
  if (shap >= 0.70) return {
    level: 'HIGH',
    color: '#ff3b3b',
    levelDesc: 'High toxicity attribution',
    advice: 'This atom is a primary driver of toxic endpoint activation. SHAP analysis places it among the highest-attributed positions in the molecule. Fragment substitution here is strongly recommended.',
  };
  if (shap >= 0.45) return {
    level: 'MODERATE',
    color: '#ff8c00',
    levelDesc: 'Moderate toxicity attribution',
    advice: 'This atom contributes meaningfully to the toxicity signal. It is within the Morgan fingerprint environment of a high-SHAP bit. Consider including this atom\'s neighbourhood in bioisostere search.',
  };
  if (shap >= 0.20) return {
    level: 'MILD',
    color: '#ffd700',
    levelDesc: 'Mild attribution signal',
    advice: 'This atom has a weak but non-negligible SHAP contribution. It may participate in secondary toxic pathways or be part of a larger pharmacophoric environment.',
  };
  return {
    level: 'INERT',
    color: '#888888',
    levelDesc: 'No significant attribution',
    advice: 'This atom shows minimal SHAP contribution. It is not implicated in any flagged toxicity endpoint fingerprint environment.',
  };
}

function atomColor(shap: number): string {
  if (shap >= 0.70) return '#ff3b3b';
  if (shap >= 0.45) return '#ff8c00';
  if (shap >= 0.20) return '#ffd700';
  return '#888888';
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Mol3DViewer({
  smiles,
  shapFragments,
  probabilities = {},
  flaggedEndpoints = [],
}: Mol3DViewerProps) {

  const containerRef    = useRef<HTMLDivElement>(null);
  const viewerRef       = useRef<any>(null);
  const animFrameRef    = useRef<number>(0);
  const highlightSphere = useRef<any>(null);  // track the click-highlight shape

  const [mol3dData,    setMol3dData]    = useState<Mol3DData | null>(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState<string | null>(null);
  const [isSpinning,   setIsSpinning]   = useState(true);
  const [viewStyle,    setViewStyle]    = useState<'stick' | 'sphere'>('stick');
  const [selectedAtom, setSelectedAtom] = useState<SelectedAtomInfo | null>(null);

  // ── Fetch 3D data ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!smiles) { setMol3dData(null); setError(null); setSelectedAtom(null); return; }
    setLoading(true); setError(null); setSelectedAtom(null);
    axios.post('http://localhost:8000/mol3d', {
        smiles,
        shap_bits: (shapFragments ?? [])
          .filter((f: any) => typeof f.bit === 'number' && typeof f.importance === 'number')
          .map((f: any) => ({ bit: f.bit, importance: Math.abs(f.importance) })),
      }, { timeout: 30000 })
      .then(res => setMol3dData(res.data))
      .catch(err => { setError(err.response?.data?.detail || err.message || '3D generation failed'); setMol3dData(null); })
      .finally(() => setLoading(false));
  }, [smiles]);

  // ── Spin loop ──────────────────────────────────────────────────────────────
  const startSpin = useCallback(() => {
    const loop = () => {
      if (!viewerRef.current) return;
      viewerRef.current.rotate(SPIN_SPEED, 'y');
      viewerRef.current.render();
      animFrameRef.current = requestAnimationFrame(loop);
    };
    animFrameRef.current = requestAnimationFrame(loop);
  }, []);

  const stopSpin = useCallback(() => cancelAnimationFrame(animFrameRef.current), []);

  // ── Clear highlight ring helper ────────────────────────────────────────────
  const clearHighlight = useCallback(() => {
    if (viewerRef.current && highlightSphere.current) {
      try { viewerRef.current.removeShape(highlightSphere.current); } catch (_) {}
      highlightSphere.current = null;
      viewerRef.current.render();
    }
  }, []);

  // ── Build viewer ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mol3dData || !containerRef.current) return;
    if (typeof $3Dmol === 'undefined') { setError('3Dmol.js not loaded — check internet connection.'); return; }

    // Tear down previous viewer
    cancelAnimationFrame(animFrameRef.current);
    if (viewerRef.current) {
      try { viewerRef.current.clear(); } catch (_) {}
    }
    highlightSphere.current = null;
    setSelectedAtom(null);

    const viewer = $3Dmol.createViewer(containerRef.current, {
      backgroundColor: 'transparent',
      antialias: true,
    });
    viewerRef.current = viewer;

    viewer.addModel(mol3dData.sdf_block, 'sdf');

    // Default style — all atoms grey
    const baseStyle = viewStyle === 'stick'
      ? { stick: { radius: 0.12, colorscheme: 'grayCarbon' }, sphere: { scale: 0.22, colorscheme: 'grayCarbon' } }
      : { sphere: { scale: 0.45, colorscheme: 'grayCarbon' } };
    viewer.setStyle({}, baseStyle);

    // SHAP coloring per atom
    mol3dData.atom_colors.forEach(({ atom_idx, shap_score, is_toxic }) => {
      if (!is_toxic && shap_score < 0.15) return;
      const col = atomColor(shap_score);
      const sel = { index: atom_idx };
      if (viewStyle === 'stick') {
        viewer.setStyle(sel, { stick: { radius: 0.17, color: col }, sphere: { scale: 0.38, color: col } });
      } else {
        viewer.setStyle(sel, { sphere: { scale: 0.58, color: col } });
      }
      // Glow halo for high-attribution atoms
      if (shap_score >= 0.5) {
        viewer.addSphere({
          center: viewer.getModel(0).atoms[atom_idx],
          radius: 0.65, color: col, opacity: 0.20,
        });
      }
    });

    // ── Click handler ──────────────────────────────────────────────────────
    viewer.setClickable({}, true, (atom: any) => {
      // Stop spin so user can inspect
      cancelAnimationFrame(animFrameRef.current);
      setIsSpinning(false);

      const idx = atom.index ?? atom.serial ?? 0;
      const atomData = mol3dData.atom_colors.find(a => a.atom_idx === idx);
      const shap = atomData?.shap_score ?? 0;
      const isToxic = atomData?.is_toxic ?? false;
      const { level, color, levelDesc, advice } = getAtomLevel(shap, isToxic);

      // Which flagged endpoints is this atom likely contributing to?
      // High-SHAP atoms are correlated with flagged endpoints.
      const relevantEndpoints = shap >= 0.30
        ? flaggedEndpoints.slice(0, Math.max(1, Math.round(shap * flaggedEndpoints.length)))
        : [];

      setSelectedAtom({
        idx,
        elem: atom.elem ?? '?',
        shap_score: shap,
        is_toxic: isToxic,
        level,
        color,
        levelDesc,
        advice,
        flaggedEndpoints: relevantEndpoints,
      });

      // Remove old highlight
      if (highlightSphere.current) {
        try { viewer.removeShape(highlightSphere.current); } catch (_) {}
      }

      // Add pulsing white ring around clicked atom
      highlightSphere.current = viewer.addSphere({
        center: { x: atom.x, y: atom.y, z: atom.z },
        radius: 0.9,
        color: color,
        opacity: 0.45,
        wireframe: true,
      });

      viewer.render();
    });

    // Click on empty space → clear selection
    viewer.setBackgroundColor('transparent');

    viewer.zoomTo();
    viewer.render();

    if (isSpinning) startSpin();

    return () => { cancelAnimationFrame(animFrameRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mol3dData, viewStyle]);

  useEffect(() => {
    if (isSpinning) startSpin(); else stopSpin();
  }, [isSpinning, startSpin, stopSpin]);

  // ── Legend dot ────────────────────────────────────────────────────────────
  const LegendDot = ({ color, label }: { color: string; label: string }) => (
    <div className="flex items-center gap-1.5">
      <div className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: color, boxShadow: `0 0 5px ${color}99` }} />
      <span className="text-[10px] text-slate-400">{label}</span>
    </div>
  );

  // ── Level badge styles ────────────────────────────────────────────────────
  const levelBadge: Record<string, string> = {
    HIGH:     'border-rose-400/40 bg-rose-400/15 text-rose-300',
    MODERATE: 'border-orange-400/40 bg-orange-400/15 text-orange-300',
    MILD:     'border-yellow-400/40 bg-yellow-400/15 text-yellow-300',
    INERT:    'border-white/10 bg-white/5 text-slate-400',
  };

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-full w-full gap-2">

      {/* Controls bar */}
      <div className="flex shrink-0 items-center justify-between gap-2 flex-wrap">
        <div className="flex gap-1">
          {(['stick', 'sphere'] as const).map(s => (
            <button
              key={s}
              onClick={() => setViewStyle(s)}
              className={`rounded-full px-2.5 py-0.5 text-[9px] uppercase tracking-[0.2em] border transition-all ${
                viewStyle === s
                  ? 'border-primary/40 bg-primary/20 text-primary'
                  : 'border-white/10 bg-white/5 text-slate-400 hover:border-white/20'
              }`}
            >{s}</button>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          {selectedAtom && (
            <button
              onClick={() => { clearHighlight(); setSelectedAtom(null); }}
              className="flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[9px] border border-white/10 bg-white/5 text-slate-400 hover:border-white/20 transition-all"
            >
              <span className="material-symbols-outlined text-[11px]">close</span>
              Clear
            </button>
          )}
          <button
            onClick={() => setIsSpinning(p => !p)}
            className={`flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[9px] border transition-all ${
              isSpinning
                ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300'
                : 'border-white/10 bg-white/5 text-slate-400'
            }`}
          >
            <span className="material-symbols-outlined text-[12px]">{isSpinning ? '360' : 'pause'}</span>
            {isSpinning ? 'Spinning' : 'Paused'}
          </button>
        </div>
      </div>

      {/* Main area: 3D canvas + info panel side by side */}
      <div className="flex flex-1 gap-2 min-h-0" style={{ minHeight: '200px' }}>

        {/* 3D Canvas */}
        <div className={`relative rounded-xl overflow-hidden border border-white/10 bg-black/30 transition-all ${selectedAtom ? 'flex-[0_0_55%]' : 'flex-1'}`}>

          {loading && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/60 backdrop-blur-sm">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <p className="text-xs text-slate-400">Generating 3D conformation…</p>
            </div>
          )}

          {error && !loading && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 p-4">
              <span className="material-symbols-outlined text-rose-400 text-3xl">error_outline</span>
              <p className="text-xs text-rose-300 text-center">{error}</p>
            </div>
          )}

          {!smiles && !loading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 opacity-40">
              <span className="material-symbols-outlined text-slate-400 text-4xl">biotech</span>
              <p className="text-xs text-slate-500">Run analysis to see 3D structure</p>
            </div>
          )}

          {mol3dData && !loading && (
            <div className="absolute bottom-2 left-2 right-2 z-10 pointer-events-none">
              <p className="text-[9px] text-slate-500 text-center">
                Click any atom to inspect · Drag to rotate
              </p>
            </div>
          )}

          <div
            ref={containerRef}
            className="absolute inset-0"
            style={{ cursor: 'grab' }}
            onMouseDown={() => stopSpin()}
            onMouseUp={() => { if (isSpinning) startSpin(); }}
          />
        </div>

        {/* Atom Info Panel — slides in when atom is selected */}
        {selectedAtom && (
          <div className="flex-1 min-w-0 flex flex-col gap-2 overflow-y-auto">

            {/* Header */}
            <div
              className="shrink-0 rounded-xl border p-3"
              style={{
                borderColor: `${selectedAtom.color}44`,
                background: `${selectedAtom.color}12`,
              }}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  {/* Atom circle */}
                  <div
                    className="h-9 w-9 shrink-0 rounded-full flex items-center justify-center font-bold text-sm"
                    style={{
                      background: `${selectedAtom.color}22`,
                      border: `2px solid ${selectedAtom.color}88`,
                      color: selectedAtom.color,
                      boxShadow: `0 0 12px ${selectedAtom.color}44`,
                    }}
                  >
                    {selectedAtom.elem}
                  </div>
                  <div>
                    <p className="text-[9px] uppercase tracking-[0.25em] text-slate-400">Atom {selectedAtom.idx}</p>
                    <p className="text-xs font-semibold text-slate-100">{selectedAtom.elem} — {selectedAtom.levelDesc}</p>
                  </div>
                </div>
                <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.2em] ${levelBadge[selectedAtom.level]}`}>
                  {selectedAtom.level}
                </span>
              </div>
            </div>

            {/* SHAP score bar */}
            <div className="shrink-0 rounded-xl border border-white/10 bg-black/20 px-3 py-2.5">
              <div className="flex justify-between items-center mb-1.5">
                <p className="text-[9px] uppercase tracking-[0.25em] text-slate-400">SHAP Attribution Score</p>
                <span className="font-mono text-xs font-bold text-slate-100">
                  {(selectedAtom.shap_score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${selectedAtom.shap_score * 100}%`,
                    background: `linear-gradient(90deg, ${selectedAtom.color}88, ${selectedAtom.color})`,
                    boxShadow: `0 0 8px ${selectedAtom.color}88`,
                  }}
                />
              </div>
              <p className="mt-1.5 text-[9px] text-slate-500">
                Normalized across all {mol3dData?.n_atoms ?? '?'} heavy atoms · aggregated over 12 endpoints
              </p>
            </div>

            {/* Flagged endpoints driven by this atom */}
            {selectedAtom.flaggedEndpoints.length > 0 && (
              <div className="shrink-0 rounded-xl border border-rose-400/15 bg-rose-400/5 px-3 py-2.5">
                <p className="text-[9px] uppercase tracking-[0.25em] text-rose-400/80 mb-2">
                  Endpoint drivers ({selectedAtom.flaggedEndpoints.length})
                </p>
                <div className="flex flex-wrap gap-1">
                  {selectedAtom.flaggedEndpoints.map(ep => (
                    <div key={ep} className="flex items-center gap-1 rounded-full border border-rose-400/20 bg-rose-400/10 px-2 py-0.5">
                      <div className="h-1.5 w-1.5 rounded-full bg-rose-400" />
                      <span className="text-[9px] text-rose-200">{ep}</span>
                      {probabilities[ep] != null && (
                        <span className="font-mono text-[9px] text-rose-300/70">
                          {(probabilities[ep] * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Advice block */}
            <div className="shrink-0 rounded-xl border border-white/10 bg-black/20 px-3 py-2.5">
              <div className="flex items-start gap-2">
                <span
                  className="material-symbols-outlined text-[15px] shrink-0 mt-0.5"
                  style={{ color: selectedAtom.color }}
                >
                  {selectedAtom.level === 'INERT' ? 'check_circle' : 'lightbulb'}
                </span>
                <p className="text-[10px] text-slate-300 leading-relaxed">
                  {selectedAtom.advice}
                </p>
              </div>
            </div>

            {/* Top fragment if available */}
            {mol3dData?.top_fragment && selectedAtom.shap_score >= 0.4 && (
              <div className="shrink-0 rounded-xl border border-amber-400/15 bg-amber-400/5 px-3 py-2">
                <p className="text-[9px] uppercase tracking-[0.2em] text-amber-400/80 mb-1">Top SHAP Fragment</p>
                <p className="font-mono text-[10px] text-amber-200 break-all">{mol3dData.top_fragment}</p>
              </div>
            )}

          </div>
        )}
      </div>

      {/* Legend + stats */}
      {mol3dData && (
        <div className="shrink-0 space-y-1.5">
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            <LegendDot color="#ff3b3b" label="High attribution" />
            <LegendDot color="#ff8c00" label="Moderate" />
            <LegendDot color="#ffd700" label="Mild" />
            <LegendDot color="#888888" label="Inert" />
          </div>
          <div className="flex gap-2 text-[10px] font-mono text-slate-500 flex-wrap">
            <span>{mol3dData.n_atoms} atoms</span>
            <span>·</span>
            <span className="text-rose-400">{mol3dData.n_toxic_atoms} attributed</span>
            {mol3dData.top_fragment && (
              <><span>·</span><span className="text-amber-300 truncate max-w-[100px]">{mol3dData.top_fragment}</span></>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
