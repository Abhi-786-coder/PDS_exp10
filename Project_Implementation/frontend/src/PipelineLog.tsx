/**
 * PipelineLog.tsx
 * Live terminal-style SSE log panel.
 * Connects to GET /logs/stream when analysis starts, streams structured log
 * events from the backend pipeline, renders them as a styled activity feed.
 */
import React, { useEffect, useRef, useState } from 'react';

interface LogEntry {
  t: number;       // elapsed seconds since pipeline start
  level: string;   // INFO | SUCCESS | WARN | ERROR | DONE
  icon: string;    // emoji icon
  msg: string;
}

interface PipelineLogProps {
  isLoading: boolean;   // true while /analyze is in-flight
}

const LEVEL_STYLE: Record<string, string> = {
  SUCCESS: 'text-emerald-300',
  WARN:    'text-amber-300',
  ERROR:   'text-rose-400',
  DONE:    'text-primary font-semibold',
  INFO:    'text-slate-300',
};

const MAX_ENTRIES = 60;

export default function PipelineLog({ isLoading }: PipelineLogProps) {
  const [entries,  setEntries]  = useState<LogEntry[]>([]);
  const [running,  setRunning]  = useState(false);
  const [lastRan,  setLastRan]  = useState<number | null>(null);
  const esRef      = useRef<EventSource | null>(null);
  const scrollRef  = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [entries]);

  // Connect/disconnect SSE when isLoading toggles
  useEffect(() => {
    if (isLoading) {
      // Reset and open stream
      setEntries([]);
      setRunning(true);

      const es = new EventSource('http://localhost:8000/logs/stream');
      esRef.current = es;

      es.onmessage = (e) => {
        try {
          const entry: LogEntry = JSON.parse(e.data);
          setEntries(prev => {
            const next = [...prev, entry];
            return next.length > MAX_ENTRIES ? next.slice(-MAX_ENTRIES) : next;
          });
          if (entry.level === 'DONE') {
            setRunning(false);
            setLastRan(Date.now());
            es.close();
          }
        } catch (_) {}
      };

      es.onerror = () => {
        setRunning(false);
        es.close();
      };

      return () => {
        es.close();
        esRef.current = null;
      };
    } else {
      // Analysis finished — let the stream close itself
      setRunning(false);
    }
  }, [isLoading]);

  const fmtTime = (t: number) => `+${t.toFixed(2)}s`;

  const isEmpty = entries.length === 0;

  return (
    <section
      style={{ gridArea: 'logs' }}
      className="glass flex flex-col rounded-2xl border border-emerald-400/10 bg-[#020d08]/60 p-4"
    >
      {/* Header */}
      <div className="flex shrink-0 items-center justify-between border-b border-white/5 pb-3 mb-3">
        <div className="flex items-center gap-3">
          {/* Terminal dot cluster */}
          <div className="flex gap-1.5">
            <div className="h-2.5 w-2.5 rounded-full bg-rose-400/70" />
            <div className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
            <div className="h-2.5 w-2.5 rounded-full bg-emerald-400/70" />
          </div>
          <div>
            <p className="text-[9px] uppercase tracking-[0.3em] text-emerald-300/60">Real-Time</p>
            <h2 className="text-sm font-semibold text-slate-100 font-mono">Pipeline Activity Log</h2>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {running && (
            <div className="flex items-center gap-1.5 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2.5 py-0.5">
              <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
              <span className="text-[9px] uppercase tracking-[0.2em] text-emerald-300">Live</span>
            </div>
          )}
          {!running && lastRan && (
            <span className="text-[9px] text-slate-500">
              Last run: {entries.length} events · {entries[entries.length - 1]?.t?.toFixed(2)}s
            </span>
          )}
          {isEmpty && !running && (
            <span className="text-[9px] text-slate-600">Waiting for analysis…</span>
          )}
        </div>
      </div>

      {/* Log body — monospace terminal */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto font-mono text-[11px] leading-relaxed space-y-0.5"
        style={{ minHeight: '120px', maxHeight: '200px' }}
      >
        {isEmpty && !running && (
          <div className="flex items-center gap-2 text-slate-600 py-4 justify-center">
            <span className="material-symbols-outlined text-[16px]">terminal</span>
            <span>Run an analysis to see the live pipeline trace here</span>
          </div>
        )}

        {entries.map((e, i) => (
          <div
            key={i}
            className={`log-entry flex items-start gap-2 px-1 py-0.5 rounded hover:bg-white/3 transition-colors`}
            style={{ animationDelay: `${Math.min(i * 0.02, 0.3)}s` }}
          >
            {/* Timestamp */}
            <span className="shrink-0 text-slate-600 w-14 text-right select-none">
              {fmtTime(e.t)}
            </span>
            {/* Icon */}
            <span className="shrink-0 select-none" style={{ fontSize: '12px', lineHeight: '18px' }}>
              {e.icon}
            </span>
            {/* Message */}
            <span className={`flex-1 break-all ${LEVEL_STYLE[e.level] ?? 'text-slate-300'}`}>
              {e.msg}
            </span>
          </div>
        ))}

        {/* Blinking cursor while running */}
        {running && (
          <div className="flex items-center gap-2 px-1 py-0.5">
            <span className="w-14 text-right text-slate-600 select-none">&nbsp;</span>
            <span className="terminal-cursor" />
          </div>
        )}
      </div>

      {/* Summary bar — only after completion */}
      {!running && entries.length > 0 && (
        <div className="shrink-0 border-t border-white/5 pt-2 mt-2 flex gap-4 flex-wrap">
          {(['SUCCESS', 'WARN', 'ERROR'] as const).map(lvl => {
            const count = entries.filter(e => e.level === lvl).length;
            if (!count) return null;
            return (
              <div key={lvl} className="flex items-center gap-1">
                <div className={`h-1.5 w-1.5 rounded-full ${
                  lvl === 'SUCCESS' ? 'bg-emerald-400' :
                  lvl === 'WARN'    ? 'bg-amber-400' :
                                      'bg-rose-400'
                }`} />
                <span className="text-[10px] text-slate-500">
                  {count} {lvl.toLowerCase()}
                </span>
              </div>
            );
          })}
          <span className="ml-auto text-[10px] text-slate-600">
            {entries.length} total events · {entries[entries.length-1]?.t?.toFixed(2) ?? '?'}s elapsed
          </span>
        </div>
      )}
    </section>
  );
}
