import { useEffect, useMemo, useRef } from "react";
import type { MlpInternalsResponse } from "../types";
import { drawGrid, formatVec, setGridCanvasSize, valueColor } from "../utils/visuals";

export type MlpInternalsConfig = {
  hiddenDim: number;
  sampleIndex: number;
  lr: number;
  seed: number;
};

type MlpInternalsPanelProps = {
  data: MlpInternalsResponse | null;
  loading: boolean;
  error: string | null;
  config: MlpInternalsConfig;
  canFetch: boolean;
  onConfigChange: (config: MlpInternalsConfig) => void;
};

function TemplateGrid({ label, grid }: { label: string; grid: number[][] }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    setGridCanvasSize(ref.current, grid.length, grid[0].length);
    drawGrid(ref.current, grid, valueColor);
  }, [grid]);

  return (
    <div className="template-card">
      <span>{label}</span>
      <canvas ref={ref} aria-label={label} />
    </div>
  );
}

export function MlpInternalsPanel({
  data,
  loading,
  error,
  config,
  canFetch,
  onConfigChange,
}: MlpInternalsPanelProps) {
  const updateConfig = (updates: Partial<MlpInternalsConfig>) => {
    onConfigChange({ ...config, ...updates });
  };

  const templatesBefore = useMemo(() => data?.hidden.templates_before ?? [], [data]);
  const templatesAfter = useMemo(() => data?.hidden.templates_after ?? [], [data]);
  const templatesGrad = useMemo(() => data?.gradients.templates ?? [], [data]);

  return (
    <section className="panel diagnostics-panel mlp-panel">
      <h2>MLP Internals</h2>
      <div className="mlp-top">
        <div className="diag-controls">
          <label>
            Hidden units
            <input
              type="number"
              min={1}
              max={8}
              value={config.hiddenDim}
              onChange={(event) => updateConfig({ hiddenDim: Number(event.target.value) })}
            />
          </label>
          <label>
            Sample idx
            <input
              type="number"
              min={0}
              value={config.sampleIndex}
              onChange={(event) => updateConfig({ sampleIndex: Number(event.target.value) })}
            />
          </label>
          <label>
            LR
            <input
              type="number"
              step="0.1"
              value={config.lr}
              onChange={(event) => updateConfig({ lr: Number(event.target.value) })}
            />
          </label>
          <label>
            Seed
            <input
              type="number"
              value={config.seed}
              onChange={(event) => updateConfig({ seed: Number(event.target.value) })}
            />
          </label>
        </div>
        <div className="mlp-math">
          <h3>What the model does</h3>
          <p><strong>Hidden:</strong> h = tanh(W₁x + b₁)</p>
          <p><strong>Output:</strong> z = W₂h + b₂</p>
          <p><strong>Probability:</strong> p̂ = σ(z)</p>
          <p><strong>Loss:</strong> L = −(y log p̂ + (1−y) log(1−p̂))</p>
          <p><strong>Update:</strong> W ← W − η ∇W L</p>
        </div>
      </div>

      {!canFetch && <p className="diag-note">Apply the custom dataset to load diagnostics.</p>}
      {error && <p className="diag-error">{error}</p>}
      {loading && <p className="diag-note">Loading internals...</p>}

      {data && (
        <>
          <div className="diag-summary">
            <div>
              <span>Input x</span>
              <strong>{formatVec(data.x)}</strong>
            </div>
            <div>
              <span>Label y</span>
              <strong>{data.y}</strong>
            </div>
            <div>
              <span>Loss</span>
              <strong>{data.loss.toFixed(4)}</strong>
            </div>
            <div>
              <span>p̂</span>
              <strong>{data.p_hat.toFixed(4)}</strong>
            </div>
          </div>

          <div className="template-section">
            <h3>Hidden weights (before)</h3>
            <p className="diag-note">
              Each template is a hidden neuron’s weight grid. Positive weights push toward +1, negatives toward -1.
            </p>
            <div className="template-grid">
              {templatesBefore.map((grid, idx) => (
                <TemplateGrid key={`before-${idx}`} label={`H${idx + 1}`} grid={grid} />
              ))}
            </div>
          </div>

          <div className="template-section">
            <h3>Hidden gradients</h3>
            <p className="diag-note">Gradients show how each weight would change to reduce loss.</p>
            <div className="template-grid">
              {templatesGrad.map((grid, idx) => (
                <TemplateGrid key={`grad-${idx}`} label={`ΔH${idx + 1}`} grid={grid} />
              ))}
            </div>
          </div>

          <div className="template-section">
            <h3>Hidden weights (after)</h3>
            <p className="diag-note">Post‑update weights after one SGD step on the selected sample.</p>
            <div className="template-grid">
              {templatesAfter.map((grid, idx) => (
                <TemplateGrid key={`after-${idx}`} label={`H${idx + 1}`} grid={grid} />
              ))}
            </div>
          </div>
        </>
      )}
      {!data && !loading && !error && canFetch && (
        <p className="diag-note">Adjust controls to inspect hidden layers.</p>
      )}
    </section>
  );
}
