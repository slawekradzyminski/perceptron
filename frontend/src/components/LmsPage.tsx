import { useEffect, useMemo, useRef, useState } from "react";
import { useLmsApi } from "../hooks/useLmsApi";
import { useHotkeys } from "../hooks/useHotkeys";
import type { CustomConfig, LmsStep } from "../types";
import { BoundaryPanel } from "./BoundaryPanel";
import { pointsForDataset } from "../utils/custom";

type LmsPageProps = {
  apiBase: string;
  datasetName: string;
  customConfig: CustomConfig;
  customApplied: boolean;
};

const OR_DATASET = [
  { x: [-1, -1], y: -1 },
  { x: [-1, 1], y: 1 },
  { x: [1, -1], y: 1 },
  { x: [1, 1], y: 1 },
];

const XOR_DATASET = [
  { x: [-1, -1], y: -1 },
  { x: [-1, 1], y: 1 },
  { x: [1, -1], y: 1 },
  { x: [1, 1], y: -1 },
];

function datasetForLms(datasetName: string, customConfig: CustomConfig) {
  if (datasetName === "xor") return XOR_DATASET;
  if (datasetName === "custom") {
    return customConfig.samples
      .map((sample) => {
        const flat = sample.grid.flat();
        if (flat.length !== 2) return null;
        return { x: [flat[0], flat[1]], y: sample.y };
      })
      .filter((row): row is { x: number[]; y: number } => row !== null);
  }
  return OR_DATASET;
}

function formatVec(values: number[]) {
  return `[${values.map((val) => val.toFixed(2)).join(", ")}]`;
}

function ErrorTrend({ history, stepCount }: { history: LmsStep[]; stepCount: number }) {
  if (history.length === 0) {
    return <p className="lms-empty">Step to see the error curve.</p>;
  }
  const points = history.map((row) => row.error * row.error);
  const maxVal = Math.max(...points, 1e-6);
  const minVal = Math.min(...points);
  const height = 90;
  const width = 260;
  const padding = 8;
  const span = Math.max(maxVal - minVal, 1e-6);
  const step = points.length > 1 ? width / (points.length - 1) : width;
  const path = points
    .map((val, idx) => {
      const x = idx * step;
      const y = height - padding - ((val - minVal) / span) * (height - padding * 2);
      return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const latest = history[history.length - 1];
  const latestE = latest.error * latest.error;
  return (
    <div className="lms-trend">
      <div className="lms-trend-header">
        <div>
          <strong>Latest step:</strong> {stepCount}
        </div>
        <div className="lms-trend-meta">
          <span>Window: {history.length}/32</span>
          <span>E={latestE.toFixed(3)}</span>
          <span>ŷ={latest.y_hat.toFixed(2)}</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Error trend">
        <defs>
          <linearGradient id="lmsTrend" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#d36a56" />
            <stop offset="100%" stopColor="#b94a3b" />
          </linearGradient>
        </defs>
        <path d={path} />
        {points.map((val, idx) => {
          const x = idx * step;
          const y = height - padding - ((val - minVal) / span) * (height - padding * 2);
          return <circle key={idx} cx={x} cy={y} r={2.2} />;
        })}
      </svg>
      <div className="lms-trend-range">E range (last {history.length} steps)</div>
      <div className="lms-trend-axis">
        <span>Min E: {minVal.toFixed(2)}</span>
        <span>Max E: {maxVal.toFixed(2)}</span>
      </div>
    </div>
  );
}

function LmsRow({
  step,
  index,
  stepCount,
  onHover,
  onLeave,
  firstStepNumber,
}: {
  step: LmsStep;
  index: number;
  stepCount: number;
  onHover: (payload: { visible: boolean; text: string; x: number; y: number }) => void;
  onLeave: () => void;
  firstStepNumber: number;
}) {
  const correct = step.y * step.y_hat > 0;
  const displayStep = firstStepNumber + index;
  const eta = step.lr;
  const [x1, x2] = step.x;
  const w1 = step.w_before[0];
  const w2 = step.w_before[1];
  const s = step.y_hat;
  const e = step.error;
  const tooltip = [
    `Step ${displayStep}`,
    `ŷ = w1*x1 + w2*x2 + b`,
    `= (${w1.toFixed(2)}*${x1.toFixed(0)}) + (${w2.toFixed(2)}*${x2.toFixed(0)}) + ${step.b_before.toFixed(2)}`,
    `= ${s.toFixed(2)}`,
    `e = y - ŷ = ${step.y.toFixed(0)} - ${s.toFixed(2)} = ${e.toFixed(2)}`,
    `Δw1 = η·e·x1 = ${eta.toFixed(2)}·${e.toFixed(2)}·${x1.toFixed(0)} = ${(eta * e * x1).toFixed(2)}`,
    `Δw2 = η·e·x2 = ${eta.toFixed(2)}·${e.toFixed(2)}·${x2.toFixed(0)} = ${(eta * e * x2).toFixed(2)}`,
    `Δb = η·e = ${eta.toFixed(2)}·${e.toFixed(2)} = ${(eta * e).toFixed(2)}`,
    `E = e² = ${(e * e).toFixed(3)}`,
  ].join("\n");
  return (
    <tr
      onMouseMove={(event) => {
        const padding = 12;
        const boxWidth = 320;
        const boxHeight = 190;
        const viewportW = window.innerWidth;
        const viewportH = window.innerHeight;
        const flipX = event.clientX + padding + boxWidth > viewportW;
        const flipY = event.clientY + padding + boxHeight > viewportH;
        const x = Math.max(8, Math.min(viewportW - boxWidth - 8, event.clientX + (flipX ? -boxWidth - padding : padding)));
        const y = Math.max(8, Math.min(viewportH - boxHeight - 8, event.clientY + (flipY ? -boxHeight - padding : padding)));
        onHover({
          visible: true,
          text: tooltip,
          x,
          y,
        });
      }}
      onMouseLeave={onLeave}
    >
      <td>{displayStep}</td>
      <td>{step.x[0].toFixed(0)}</td>
      <td>{step.x[1].toFixed(0)}</td>
      <td>1</td>
      <td>{step.y.toFixed(0)}</td>
      <td>{step.w_before[0].toFixed(2)}</td>
      <td>{step.w_before[1].toFixed(2)}</td>
      <td>{step.b_before.toFixed(2)}</td>
      <td>{step.y_hat.toFixed(2)}</td>
      <td className={correct ? "ok" : "bad"}>{correct ? "✓" : "✗"}</td>
      <td>{step.w_after[0].toFixed(2)}</td>
      <td>{step.w_after[1].toFixed(2)}</td>
      <td>{step.b_after.toFixed(2)}</td>
      <td>{step.grad_w1.toFixed(2)}</td>
      <td>{step.grad_w2.toFixed(2)}</td>
      <td>{step.grad_b.toFixed(2)}</td>
    </tr>
  );
}

export function LmsPage({ apiBase, datasetName, customConfig, customApplied }: LmsPageProps) {
  const { state, history, stepCount, error, loading, step, reset, resetWithOptions } = useLmsApi(apiBase);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ visible: boolean; text: string; x: number; y: number }>({
    visible: false,
    text: "",
    x: 0,
    y: 0,
  });
  const [lmsLr, setLmsLr] = useState(0.1);
  const datasetRows = useMemo(() => datasetForLms(datasetName, customConfig), [datasetName, customConfig]);
  const datasetDim = datasetName === "custom" ? customConfig.rows * customConfig.cols : 2;
  const canUseCustom = datasetName !== "custom" || (customApplied && datasetDim === 2);
  const points = useMemo(
    () =>
      pointsForDataset(datasetName, datasetDim, customConfig),
    [datasetName, datasetDim, customConfig],
  );
  const w = state?.w ?? [0, 0];
  const b = state?.b ?? 0;
  const firstStepNumber = stepCount > history.length ? stepCount - history.length + 1 : 1;

  useEffect(() => {
    if (!canUseCustom) {
      return;
    }
    void resetWithOptions({ datasetName, customConfig, customApplied, lr: lmsLr });
  }, [resetWithOptions, datasetName, customConfig, customApplied, canUseCustom, lmsLr]);

  useHotkeys({ onStep: step, onReset: reset, enabled: canUseCustom });

  return (
    <section className="panel lms-panel">
      <div className="lms-head">
        <div>
          <h2>LMS (Widrow–Hoff) Exercise</h2>
          <p className="panel-subtle">
            We minimize squared error by nudging weights in the direction of the gradient.
          </p>
        </div>
        <div className="lms-actions">
          <button type="button" onClick={step} disabled={loading || !canUseCustom}>
            Step
          </button>
          <button type="button" onClick={reset} disabled={loading || !canUseCustom}>
            Reset
          </button>
        </div>
      </div>

      <div className="lms-grid">
        <BoundaryPanel w={w} b={b} points={points} show={true} canvasRef={canvasRef} />
        <div className="lms-math">
          <h3>Update rule</h3>
          <p><strong>Prediction:</strong> ŷ = w·x + b</p>
          <p><strong>Error:</strong> e = y − ŷ</p>
          <p><strong>Update:</strong> w ← w + η e x, b ← b + η e</p>
          <p><strong>Gradient:</strong> ∂E/∂wᵢ = −2 e xᵢ, ∂E/∂b = −2 e</p>
        </div>
        <div className="lms-math">
          <h3>Dataset</h3>
          <p><strong>Active:</strong> {datasetName.toUpperCase()}</p>
          <label className="lms-label" aria-label="Learning rate">
            <span>Learning rate (η)</span>
            <input
              type="number"
              min="0.01"
              step="0.05"
              value={lmsLr}
              onChange={(event) => setLmsLr(Number(event.target.value))}
              disabled={!canUseCustom}
            />
          </label>
          {datasetName === "custom" && datasetDim !== 2 && (
            <p className="diag-note">LMS supports only 2D custom datasets (1×2 grid).</p>
          )}
          <ul>
            {datasetRows.map((row, idx) => (
              <li key={idx}>
                x = [{row.x[0]}, {row.x[1]}], y = {row.y}
              </li>
            ))}
          </ul>
          {state && (
            <>
              <p><strong>Current w:</strong> {formatVec(state.w)}</p>
              <p><strong>Current b:</strong> {state.b.toFixed(2)}</p>
              <p><strong>Learning rate:</strong> {state.lr.toFixed(2)}</p>
              <p><strong>Sample idx:</strong> {state.idx} / {state.sample_count - 1}</p>
            </>
          )}
        </div>
        <div className="lms-math">
          <h3>Error trend (E = (y − ŷ)²)</h3>
          <ErrorTrend history={history} stepCount={stepCount} />
        </div>
      </div>

      {error && <p className="diag-error">{error}</p>}

      <div className="lms-table-wrap">
        <table className="lms-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>x₁</th>
              <th>x₂</th>
              <th>x_b</th>
              <th>y</th>
              <th>w₁</th>
              <th>w₂</th>
              <th>b</th>
              <th>ŷ</th>
              <th>✓?</th>
              <th>w₁′</th>
              <th>w₂′</th>
              <th>b′</th>
              <th>∂E/∂w₁</th>
              <th>∂E/∂w₂</th>
              <th>∂E/∂b</th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 && (
              <tr>
                <td colSpan={16} className="lms-empty">
                  Click Step to start filling the LMS table.
                </td>
              </tr>
            )}
            {history.map((row, index) => (
              <LmsRow
                key={`${row.idx}-${index}`}
                step={row}
                index={index}
                stepCount={stepCount}
                firstStepNumber={firstStepNumber}
                onHover={(payload) => setTooltip(payload)}
                onLeave={() => setTooltip((prev) => ({ ...prev, visible: false }))}
              />
            ))}
          </tbody>
        </table>
      </div>
      <div
        className={`lms-tooltip ${tooltip.visible ? "visible" : ""}`}
        style={{ left: tooltip.x, top: tooltip.y }}
      >
        {tooltip.text}
      </div>
    </section>
  );
}
