import React, { useState, useEffect } from 'react';

// ── Confidence Waterfall ────────────────────────────────────────────────────────

function ConfidenceWaterfall({ chain }: { chain: any }) {
  if (!chain || !chain.steps) return null;

  const { steps, final_confidence, verdict, verdict_level } = chain;
  const MAX_BAR = 40; // max % of container width for delta bars

  // Find the largest absolute delta for normalization
  const maxDelta = Math.max(...steps.map((s: any) => Math.abs(s.delta || 0)), 1);

  const verdictColors: Record<string, string> = {
    danger:  '#ef4444',
    warn:    '#f59e0b',
    neutral: '#6366f1',
    good:    '#22c55e',
    safe:    '#10b981',
  };
  const verdictColor = verdictColors[verdict_level] || '#6366f1';

  return (
    <div className="confidence-waterfall">
      {steps.map((step: any, idx: number) => {
        const isBase    = step.direction === 'base';
        const isPos     = step.direction === 'positive';
        const isNeg     = step.direction === 'negative';
        const barWidth  = isBase ? 0 : Math.round((Math.abs(step.delta) / maxDelta) * MAX_BAR);
        const barColor  = isBase ? '#3b82f6' : isPos ? '#22c55e' : isNeg ? '#ef4444' : '#6b7280';

        return (
          <div key={idx} className="wf-row">
            {/* Step label */}
            <div className="wf-label">
              <span className="wf-icon">{step.icon}</span>
              <span className="wf-name">{step.name}</span>
            </div>

            {/* Bar track */}
            <div className="wf-track">
              {isBase ? (
                <div
                  className="wf-bar wf-base"
                  style={{ width: `${Math.round((step.running_total / 100) * MAX_BAR)}%` }}
                />
              ) : (
                <div
                  className={`wf-bar ${isPos ? 'wf-pos' : isNeg ? 'wf-neg' : 'wf-neutral'}`}
                  style={{ width: `${barWidth}%` }}
                />
              )}
              <span className="wf-delta">
                {isBase ? `${step.running_total}%` : step.delta > 0 ? `+${step.delta}%` : step.delta < 0 ? `${step.delta}%` : '±0%'}
              </span>
              <span className="wf-running">→ {step.running_total}%</span>
            </div>

            {/* Description tooltip row */}
            <div className="wf-desc">{step.description}</div>
          </div>
        );
      })}

      {/* Final verdict */}
      <div className="wf-verdict" style={{ borderColor: verdictColor }}>
        <div className="wf-verdict-score" style={{ color: verdictColor }}>
          {final_confidence}%
        </div>
        <div className="wf-verdict-label" style={{ color: verdictColor }}>
          {verdict}
        </div>
        <div className="wf-verdict-sub">
          Combined confidence across all 9 pipeline signals
        </div>
      </div>
    </div>
  );
}

interface DecisionTraceProps {
  isOpen: boolean;
  onClose: () => void;
}

export function DecisionTrace({ isOpen, onClose }: DecisionTraceProps) {
  const [traceData, setTraceData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      setError(null);
      fetch('http://localhost:8000/trace')
        .then(res => res.json())
        .then(data => {
          if (data.error) {
            setError(data.error);
          } else {
            setTraceData(data);
          }
          setLoading(false);
        })
        .catch(err => {
          setError(err.toString());
          setLoading(false);
        });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="trace-overlay">
      <div className="trace-modal">
        <header className="trace-header">
          <h2>
            <span className="material-symbols-outlined">plumbing</span>
            Pipeline Decision Trace & Audit Log
          </h2>
          <button className="close-btn" onClick={onClose}>
            <span className="material-symbols-outlined">close</span>
          </button>
        </header>

        <div className="trace-content">
          {loading && <div className="trace-loading">Fetching audit log...</div>}
          {error && <div className="trace-error">{error}</div>}
          
          {!loading && !error && traceData && (
            <div className="trace-sections">
              
              {/* META INFO */}
              <div className="trace-meta">
                <span className="badge">Analyzed: {traceData.timestamp}</span>
                <span className="badge">Processing Time: {traceData.elapsed_s}s</span>
                <div className="smiles-break">{traceData.smiles}</div>
              </div>

              {/* 0. CONFIDENCE CHAIN */}
              {traceData.confidence_chain && (
                <section className="trace-card trace-card-highlight">
                  <h3>0. Decision Confidence Chain</h3>
                  <p className="trace-card-sub">
                    How each pipeline layer contributes to the final toxicity confidence score.
                    Each step adds or subtracts based on real signal strength.
                  </p>
                  <ConfidenceWaterfall chain={traceData.confidence_chain} />
                </section>
              )}

              {/* 1. MOLECULE PROPS */}
              <section className="trace-card">
                <h3>1. Molecular Properties & Lipinski</h3>
                <div className="trace-grid">
                  <div className="stat"><span>MW</span><strong>{traceData.mol_properties.mw}</strong></div>
                  <div className="stat"><span>LogP</span><strong>{traceData.mol_properties.logp}</strong></div>
                  <div className="stat"><span>HBA</span><strong>{traceData.mol_properties.hba}</strong></div>
                  <div className="stat"><span>HBD</span><strong>{traceData.mol_properties.hbd}</strong></div>
                  <div className="stat"><span>TPSA</span><strong>{traceData.mol_properties.tpsa}</strong></div>
                  <div className="stat"><span>Lipinski</span>
                    <strong className={traceData.mol_properties.lipinski_ok ? 'pass' : 'fail'}>
                      {traceData.mol_properties.lipinski_ok ? 'PASS' : 'FAIL'}
                    </strong>
                  </div>
                </div>
              </section>

              {/* 2. OOD */}
              <section className="trace-card">
                <h3>2. Applicability Domain (OOD)</h3>
                <div className={`status-banner ${traceData.ood.in_domain ? 'good' : 'warn'}`}>
                  <strong>{traceData.ood.in_domain ? 'IN-DOMAIN' : 'OUT-OF-DOMAIN'}</strong>
                  <p>{traceData.ood.message}</p>
                </div>
              </section>

              {/* 3. ENDPOINTS */}
              <section className="trace-card">
                <h3>3. Model Predictions & Imbalance Metrics</h3>
                <table className="trace-table">
                  <thead>
                    <tr>
                      <th>Endpoint</th>
                      <th>Probability</th>
                      <th>Threshold</th>
                      <th>Margin</th>
                      <th>Class Imbalance</th>
                      <th>Decision</th>
                    </tr>
                  </thead>
                  <tbody>
                    {traceData.endpoints.map((ep: any) => (
                      <tr key={ep.endpoint}>
                        <td>{ep.endpoint}</td>
                        <td className={ep.prob > ep.threshold ? 'warn-text' : 'good-text'}>{ep.prob.toFixed(4)}</td>
                        <td>{ep.threshold.toFixed(4)}</td>
                        <td>±{ep.margin.toFixed(4)}</td>
                        <td>1:{ep.imbalance}</td>
                        <td><span className={`label ${ep.decision.toLowerCase()}`}>{ep.decision}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>

              {/* 4. CONSTELLATION */}
              <section className="trace-card">
                <h3>4. Toxicity Constellation (12D Response Mapping)</h3>
                {traceData.constellation?.name ? (
                  <div className="trace-list">
                    <p><strong>Classification:</strong> {traceData.constellation.name}</p>
                    <p><strong>Proximity:</strong> {traceData.constellation.proximity} (distance: {traceData.constellation.distance})</p>
                    <p><strong>Mechanism Hint:</strong> {traceData.constellation.mechanism}</p>
                  </div>
                ) : (
                  <p className="dim">Constellation data unavailable.</p>
                )}
              </section>

              {/* 5. STBI */}
              <section className="trace-card">
                <h3>5. Scaffold Toxicity Brittleness Index (STBI)</h3>
                {traceData.stbi?.scaffold ? (
                  <div className="trace-list">
                    <p><strong>Murcko Scaffold:</strong> {traceData.stbi.scaffold}</p>
                    <p><strong>Brittleness Score:</strong> {traceData.stbi.score ? traceData.stbi.score.toFixed(4) : 'N/A'}</p>
                    <p><strong>Assessment:</strong> <span className={`label ${traceData.stbi.assessment.toLowerCase()}`}>{traceData.stbi.assessment}</span></p>
                    <p><strong>Note:</strong> {traceData.stbi.message}</p>
                  </div>
                ) : (
                  <p className="dim">STBI data unavailable.</p>
                )}
              </section>

              {/* 6. MTEP */}
              <section className="trace-card">
                <h3>6. Minimum Toxicity Escape Path (MTEP)</h3>
                {traceData.mtep_summary?.prognosis ? (
                  <>
                    <div className="trace-grid mtep-summary">
                      <div className="stat"><span>Prognosis</span><strong>{traceData.mtep_summary.prognosis}</strong></div>
                      <div className="stat"><span>EASY Escapes</span><strong>{traceData.mtep_summary.n_easy}</strong></div>
                      <div className="stat"><span>HARD Escapes</span><strong>{traceData.mtep_summary.n_hard}</strong></div>
                      <div className="stat"><span>TRAPPED</span><strong>{traceData.mtep_summary.n_trapped}</strong></div>
                    </div>
                    <table className="trace-table mt-4">
                      <thead>
                        <tr>
                          <th>Endpoint</th>
                          <th>Gradient Norm</th>
                          <th>Difficulty</th>
                        </tr>
                      </thead>
                      <tbody>
                        {traceData.mtep.map((ep: any) => (
                          <tr key={ep.endpoint}>
                            <td>{ep.endpoint}</td>
                            <td>{ep.gradient_norm.toFixed(5)}</td>
                            <td><span className={`label ${ep.difficulty.toLowerCase()}`}>{ep.difficulty}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                ) : (
                  <p className="dim">MTEP data unavailable.</p>
                )}
              </section>

              {/* 7. SHAP */}
              <section className="trace-card">
                <h3>7. SHAP Fragment Attribution</h3>
                <table className="trace-table">
                  <thead>
                    <tr>
                      <th>Fragment (SMILES)</th>
                      <th>Morgan Bit</th>
                      <th>Importance (Sum)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {traceData.shap.map((sh: any, i: number) => (
                      <tr key={i}>
                        <td className="smiles-font">{sh.fragment || 'N/A (Bit only)'}</td>
                        <td>{sh.bit}</td>
                        <td>+{sh.importance.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>

              {/* 8. CANDIDATES */}
              <section className="trace-card">
                <h3>8. Candidate Pipeline</h3>
                {traceData.candidates ? (
                  <>
                    <div className="trace-grid">
                      <div className="stat"><span>Queried Frag</span><strong className="smiles-font">{traceData.candidates.fragment_queried}</strong></div>
                      <div className="stat"><span>ChEMBL Found</span><strong>{traceData.candidates.chembl_found}</strong></div>
                      <div className="stat"><span>ADME Passed</span><strong>{traceData.candidates.after_adme}</strong></div>
                      <div className="stat"><span>Final Candidates</span><strong>{traceData.candidates.final_candidates}</strong></div>
                    </div>
                    <div className="trace-scroll-table mt-4">
                      <table className="trace-table">
                        <thead>
                          <tr>
                            <th>Rank</th>
                            <th>ID / SMILES</th>
                            <th>Δ Tox</th>
                            <th>Δ LogP</th>
                            <th>Δ MW</th>
                            <th>Synth (SAS/SCS)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {traceData.candidates.candidates?.map((c: any) => (
                            <tr key={c.chembl_id} className={c.pareto_dominated ? 'dim-row' : ''}>
                              <td>#{c.rank}</td>
                              <td>
                                <div className="text-xs text-blue-400">{c.chembl_id}</div>
                                <div className="smiles-font text-xs">{c.smiles}</div>
                              </td>
                              <td className="good-text">{c.mean_tox_delta?.toFixed(3)}</td>
                              <td>{c.delta_logp?.toFixed(2)}</td>
                              <td>{c.delta_mw?.toFixed(1)}</td>
                              <td>{c.synth_verdict} ({c.sa_score}/{c.sc_score})</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                ) : (
                  <p className="dim">No candidates processed.</p>
                )}
              </section>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}
