import { useEffect } from "react";
import { useLmsApi } from "../hooks/useLmsApi";
import { useHotkeys } from "../hooks/useHotkeys";
import type { LmsStep } from "../types";

type LmsPageProps = {
  apiBase: string;
};

const DATASET = [
  { x: [-1, -1], y: -1 },
  { x: [-1, 1], y: -1 },
  { x: [1, -1], y: 1 },
  { x: [1, 1], y: 1 },
];

function formatVec(values: number[]) {
  return `[${values.map((val) => val.toFixed(2)).join(", ")}]`;
}

function ErrorTrend({ history, stepCount }: { history: LmsStep[]; stepCount: number }) {
  if (history.length === 0) {
    return <p className="lms-empty">Step to see the error curve.</p>;
  }
  const points = [...history].reverse().map((row) => row.error * row.error);
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
  const latest = history[0];
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
      <div className="lms-trend-axis">
        <span>{minVal.toFixed(2)}</span>
        <span>{maxVal.toFixed(2)}</span>
      </div>
    </div>
  );
}

function LmsRow({ step, index, stepCount }: { step: LmsStep; index: number; stepCount: number }) {
  const correct = step.y * step.y_hat > 0;
  const displayStep = stepCount > 0 ? Math.max(stepCount - index, 1) : index + 1;
  return (
    <tr>
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

export function LmsPage({ apiBase }: LmsPageProps) {
  const { state, history, stepCount, error, loading, loadState, step, reset } = useLmsApi(apiBase);

  useEffect(() => {
    void loadState();
  }, [loadState]);

  useHotkeys({ onStep: step, onReset: reset, enabled: true });

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
          <button type="button" onClick={step} disabled={loading}>
            Step
          </button>
          <button type="button" onClick={reset} disabled={loading}>
            Reset
          </button>
        </div>
      </div>

      <div className="lms-grid">
        <div className="lms-math">
          <h3>Update rule</h3>
          <p><strong>Prediction:</strong> ŷ = w·x + b</p>
          <p><strong>Error:</strong> e = y − ŷ</p>
          <p><strong>Update:</strong> w ← w + η e x, b ← b + η e</p>
          <p><strong>Gradient:</strong> ∂E/∂wᵢ = −2 e xᵢ, ∂E/∂b = −2 e</p>
        </div>
        <div className="lms-math">
          <h3>Dataset</h3>
          <ul>
            {DATASET.map((row, idx) => (
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
              <LmsRow key={`${row.idx}-${index}`} step={row} index={index} stepCount={stepCount} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
