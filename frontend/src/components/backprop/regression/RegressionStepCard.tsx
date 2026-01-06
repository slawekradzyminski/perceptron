import type { RegressionState, RegressionStep } from "../../../types";
import { formatNum } from "../common/formatters";

type RegressionStepCardProps = {
  state: RegressionState | null;
  step: RegressionStep | null;
};

export function RegressionStepCard({ state, step }: RegressionStepCardProps) {
  const sample = step?.sample ?? state?.next_sample ?? null;
  if (!sample) {
    return (
      <div className="panel backprop-card">
        <h3>Regression Step</h3>
        <p className="diag-placeholder">Step regression to view prediction, gradients, and updates.</p>
      </div>
    );
  }

  return (
    <div className="panel backprop-card">
      <h3>Regression Step</h3>
      <div className="backprop-stack">
        <div>
          <span className="backprop-label">Sample</span>
          <div className="backprop-mono">
            x={formatNum(sample.x, 3)}, y={formatNum(sample.y, 3)}
          </div>
        </div>
        <div className="backprop-split">
          <div>
            <span className="backprop-label">ŷ</span>
            <div className="backprop-mono">{formatNum(step?.y_hat ?? 0, 4)}</div>
          </div>
          <div>
            <span className="backprop-label">Loss</span>
            <div className="backprop-mono">{formatNum(step?.loss_value ?? 0, 4)}</div>
          </div>
        </div>
        <div>
          <span className="backprop-label">Params</span>
          <div className="backprop-mono">
            m={formatNum(state?.params.m ?? 0, 4)} b={formatNum(state?.params.b ?? 0, 4)}
          </div>
        </div>
        <div>
          <span className="backprop-label">Gradients</span>
          <div className="backprop-mono">
            dm={formatNum(step?.grads.m ?? 0, 4)} db={formatNum(step?.grads.b ?? 0, 4)}
          </div>
        </div>
        <div>
          <span className="backprop-label">Mean loss</span>
          <div className="backprop-mono">{formatNum(step?.mean_loss ?? 0, 4)}</div>
        </div>
      </div>
    </div>
  );
}
