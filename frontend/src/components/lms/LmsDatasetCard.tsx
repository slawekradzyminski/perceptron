import type { LmsState } from "../../types";

export function LmsDatasetCard({
  datasetName,
  datasetDim,
  datasetRows,
  canUseCustom,
  lmsLr,
  onLrChange,
  state,
  formatVec,
}: {
  datasetName: string;
  datasetDim: number;
  datasetRows: { x: number[]; y: number }[];
  canUseCustom: boolean;
  lmsLr: number;
  onLrChange: (value: number) => void;
  state: LmsState | null;
  formatVec: (values: number[]) => string;
}) {
  return (
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
          onChange={(event) => onLrChange(Number(event.target.value))}
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
  );
}
