type RegressionControlsProps = {
  loss: "mse" | "l1";
  lr: number;
  loading: boolean;
  hasState: boolean;
  onLossChange: (value: "mse" | "l1") => void;
  onLrChange: (value: number) => void;
  onLrCommit?: () => void;
  onReset: () => void;
};

export function RegressionControls({
  loss,
  lr,
  loading,
  hasState,
  onLossChange,
  onLrChange,
  onLrCommit,
  onReset,
}: RegressionControlsProps) {
  return (
    <div className="panel backprop-card">
      <h3>Regression Controls</h3>
      <div className="backprop-controls">
        <label>
          <span>Loss</span>
          <select value={loss} onChange={(event) => onLossChange(event.target.value as "mse" | "l1")}>
            <option value="mse">MSE</option>
            <option value="l1">L1</option>
          </select>
        </label>
        <label>
          <span>Learning rate</span>
          <div className="lr-row">
            <input
              type="range"
              min="0.01"
              max="1"
              step="0.01"
              value={lr}
              onChange={(event) => onLrChange(Number(event.target.value))}
              onMouseUp={() => onLrCommit?.()}
            />
            <span>{lr.toFixed(2)}</span>
          </div>
        </label>
        <button type="button" className="ghost" onClick={onReset} disabled={loading || !hasState}>
          Reset Regression
        </button>
      </div>
    </div>
  );
}
