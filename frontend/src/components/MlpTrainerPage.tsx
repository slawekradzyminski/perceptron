import { useEffect, useMemo, useState } from "react";
import type { CustomConfig, MlpInternalsResponse, MlpTrainerSnapshot } from "../types";
import { useMlpTrainerApi } from "../hooks/useMlpTrainerApi";
import { TemplateGrid } from "./TemplateGrid";
import { formatVec } from "../utils/visuals";
import { useHotkeys } from "../hooks/useHotkeys";
import { MlpBoundaryPanel } from "./MlpBoundaryPanel";

export type MlpTrainerPageProps = {
  apiBase: string;
  datasetName: string;
  customConfig: CustomConfig;
  customApplied: boolean;
};

function EvalRow({ row }: { row: MlpTrainerSnapshot["evals"][number] }) {
  const predLabel = row.pred === 1 ? "+1" : "-1";
  return (
    <div className="mlp-eval-row">
      <span>x={formatVec(row.x)}</span>
      <span>y={row.y}</span>
      <span>p̂={row.p_hat.toFixed(3)}</span>
      <span>ŷ={predLabel}</span>
    </div>
  );
}

function StepTemplates({ step }: { step: MlpInternalsResponse }) {
  return (
    <div className="template-section">
      <h3>Last step templates</h3>
      <p className="diag-note">Before, gradients, and after update for each hidden unit.</p>
      <div className="template-grid">
        {step.hidden.templates_before.map((grid, idx) => (
          <TemplateGrid key={`step-before-${idx}`} label={`H${idx + 1} (before)`} grid={grid} />
        ))}
        {step.gradients.templates.map((grid, idx) => (
          <TemplateGrid key={`step-grad-${idx}`} label={`ΔH${idx + 1}`} grid={grid} />
        ))}
        {step.hidden.templates_after.map((grid, idx) => (
          <TemplateGrid key={`step-after-${idx}`} label={`H${idx + 1} (after)`} grid={grid} />
        ))}
      </div>
    </div>
  );
}

function WeightList({
  snapshot,
}: {
  snapshot: MlpTrainerSnapshot;
}) {
  const w1 = snapshot.hidden.weights[0] ?? [];
  const w2 = snapshot.hidden.weights[1] ?? [];
  return (
    <div className="mlp-weights">
      <h3>Weights</h3>
      <div className="mlp-weight-grid">
        <div className="mlp-weight-row">
          <span>w₁₁¹</span>
          <strong>{w1[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₁₂¹</span>
          <strong>{w1[1]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₂₁¹</span>
          <strong>{w2[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₂₂¹</span>
          <strong>{w2[1]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>b₁¹</span>
          <strong>{snapshot.hidden.bias[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>b₂¹</span>
          <strong>{snapshot.hidden.bias[1]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₁²</span>
          <strong>{snapshot.output.weights[0]?.[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₂²</span>
          <strong>{snapshot.output.weights[0]?.[1]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>b¹²</span>
          <strong>{snapshot.output.bias[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
      </div>
    </div>
  );
}

function NetworkDiagram({ snapshot }: { snapshot: MlpTrainerSnapshot }) {
  const w1 = snapshot.hidden.weights[0] ?? [0, 0];
  const w2 = snapshot.hidden.weights[1] ?? [0, 0];
  const b1 = snapshot.hidden.bias[0] ?? 0;
  const b2 = snapshot.hidden.bias[1] ?? 0;
  const wOut = snapshot.output.weights[0] ?? [0, 0];
  const bOut = snapshot.output.bias[0] ?? 0;

  return (
    <div className="mlp-diagram">
      <h3>Network diagram</h3>
      <svg viewBox="0 0 520 260" role="img" aria-label="MLP diagram">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="#5a4733" />
          </marker>
        </defs>
        <circle cx="60" cy="60" r="26" className="mlp-node input" />
        <text x="60" y="65" textAnchor="middle">x₁</text>
        <circle cx="60" cy="160" r="26" className="mlp-node input" />
        <text x="60" y="165" textAnchor="middle">x₂</text>

        <circle cx="220" cy="60" r="30" className="mlp-node hidden" />
        <text x="220" y="65" textAnchor="middle">h₁</text>
        <circle cx="220" cy="160" r="30" className="mlp-node hidden" />
        <text x="220" y="165" textAnchor="middle">h₂</text>

        <circle cx="380" cy="110" r="32" className="mlp-node output" />
        <text x="380" y="115" textAnchor="middle">ŷ</text>

        <line x1="86" y1="60" x2="190" y2="60" markerEnd="url(#arrow)" />
        <line x1="86" y1="60" x2="190" y2="160" markerEnd="url(#arrow)" />
        <line x1="86" y1="160" x2="190" y2="60" markerEnd="url(#arrow)" />
        <line x1="86" y1="160" x2="190" y2="160" markerEnd="url(#arrow)" />

        <line x1="250" y1="60" x2="348" y2="110" markerEnd="url(#arrow)" />
        <line x1="250" y1="160" x2="348" y2="110" markerEnd="url(#arrow)" />

        <text x="125" y="45" className="mlp-weight-text">w₁₁¹={w1[0].toFixed(2)}</text>
        <text x="135" y="140" className="mlp-weight-text">w₁₂¹={w1[1].toFixed(2)}</text>
        <text x="120" y="110" className="mlp-weight-text">w₂₁¹={w2[0].toFixed(2)}</text>
        <text x="125" y="200" className="mlp-weight-text">w₂₂¹={w2[1].toFixed(2)}</text>

        <text x="270" y="45" className="mlp-weight-text">w₁²={wOut[0].toFixed(2)}</text>
        <text x="270" y="175" className="mlp-weight-text">w₂²={wOut[1].toFixed(2)}</text>

        <text x="210" y="20" className="mlp-bias-text">b₁¹={b1.toFixed(2)}</text>
        <text x="210" y="230" className="mlp-bias-text">b₂¹={b2.toFixed(2)}</text>
        <text x="380" y="40" className="mlp-bias-text">b¹²={bOut.toFixed(2)}</text>
      </svg>
    </div>
  );
}

export function MlpTrainerPage({ apiBase, datasetName, customConfig, customApplied }: MlpTrainerPageProps) {
  const { snapshot, lastStep, error, loading, resetWithOptions, step } = useMlpTrainerApi(apiBase);
  const [hiddenDim, setHiddenDim] = useState(2);
  const [mlpLr, setMlpLr] = useState(0.5);
  const [seed, setSeed] = useState(0);

  const canUseCustom = datasetName !== "custom" || customApplied;

  useEffect(() => {
    if (!canUseCustom) return;
    void resetWithOptions({
      datasetName,
      customConfig,
      customApplied,
      hiddenDim,
      lr: mlpLr,
      seed,
    });
  }, [canUseCustom, datasetName, customConfig, customApplied, hiddenDim, mlpLr, seed, resetWithOptions]);

  const handleStep = () => {
    if (!canUseCustom) return;
    void step();
  };
  const handleReset = () => {
    if (!canUseCustom) return;
    void resetWithOptions({ datasetName, customConfig, customApplied, hiddenDim, lr: mlpLr, seed });
  };

  useHotkeys({ onStep: handleStep, onReset: handleReset, enabled: canUseCustom });

  const evalSummary = useMemo(() => {
    if (!snapshot) return { correct: 0, total: 0 };
    const correct = snapshot.evals.filter((row) => row.pred === row.y).length;
    return { correct, total: snapshot.evals.length };
  }, [snapshot]);

  return (
    <section className="panel mlp-train">
      <div className="mlp-grid">
        <div className="mlp-controls">
          <div className="diag-controls compact">
            <label>
              Hidden units
              <input
                type="number"
                min={1}
                max={8}
                value={hiddenDim}
                onChange={(event) => setHiddenDim(Number(event.target.value))}
              />
            </label>
            <label>
              LR
              <input
                type="number"
                step="0.05"
                min={0.05}
                value={mlpLr}
                onChange={(event) => setMlpLr(Number(event.target.value))}
              />
            </label>
            <label>
              Seed
              <input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
            </label>
            <div className="mlp-actions">
              <button type="button" onClick={handleStep} disabled={loading || !canUseCustom}>
                Step
              </button>
              <button type="button" onClick={handleReset} disabled={loading || !canUseCustom}>
                Reset
              </button>
            </div>
          </div>
        </div>

        <MlpBoundaryPanel snapshot={snapshot} />
        {snapshot && <NetworkDiagram snapshot={snapshot} />}
        <div className="mlp-side">
          {snapshot && (
            <div className="mlp-summary">
              <div>
                <span>Dataset</span>
                <strong>{snapshot.dataset.toUpperCase()}</strong>
              </div>
              <div>
                <span>Sample</span>
                <strong>{formatVec(snapshot.next_x)}</strong>
              </div>
              <div>
                <span>Target y</span>
                <strong>{snapshot.next_y}</strong>
              </div>
              <div>
                <span>Accuracy</span>
                <strong>
                  {evalSummary.correct}/{evalSummary.total}
                </strong>
              </div>
            </div>
          )}
          <div className="mlp-math">
            <h3>MLP (2-layer) training</h3>
            <p><strong>Hidden:</strong> h = tanh(W₁x + b₁)</p>
            <p><strong>Output:</strong> z = W₂h + b₂</p>
            <p><strong>Probability:</strong> p̂ = σ(z)</p>
            <p><strong>Loss:</strong> L = −(y log p̂ + (1−y) log(1−p̂))</p>
            <p><strong>Update:</strong> W ← W − η ∇W L</p>
          </div>
          {snapshot && <WeightList snapshot={snapshot} />}
        </div>

        <div className="mlp-templates">
          {snapshot && (
            <div className="template-section">
              <h3>Current hidden templates</h3>
              <div className="template-grid">
                {snapshot.hidden.templates.map((grid, idx) => (
                  <TemplateGrid key={`current-${idx}`} label={`H${idx + 1}`} grid={grid} />
                ))}
              </div>
            </div>
          )}
          {lastStep && <StepTemplates step={lastStep} />}
        </div>

        <div className="mlp-preds">
          {snapshot && (
            <div className="mlp-eval">
              <h3>Predictions on dataset</h3>
              <div className="mlp-eval-grid">
                {snapshot.evals.map((row, idx) => (
                  <EvalRow key={idx} row={row} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {!canUseCustom && <p className="diag-note">Apply the custom dataset to train the MLP.</p>}
      {error && <p className="diag-error">{error}</p>}
    </section>
  );
}
