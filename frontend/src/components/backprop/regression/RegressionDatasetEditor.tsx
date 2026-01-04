import { useMemo, useState } from "react";

type Sample = { x: number; y: number };

type RegressionDatasetEditorProps = {
  samples: Sample[];
  onApply: (samples: Sample[]) => void;
};

function parseSampleInput(value: string) {
  return value
    .split(/\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split(/[,\s]+/).map((v) => Number(v)))
    .filter((row) => row.length >= 2)
    .map(([x, y]) => ({ x, y }));
}

export function RegressionDatasetEditor({ samples, onApply }: RegressionDatasetEditorProps) {
  const [raw, setRaw] = useState(samples.map((s) => `${s.x}, ${s.y}`).join("\n"));

  const parsed = useMemo(() => parseSampleInput(raw), [raw]);

  return (
    <div className="panel backprop-card">
      <h3>Dataset Editor</h3>
      <p className="panel-subtle">Paste x,y pairs (one per line) to match the exercise data.</p>
      <div className="backprop-controls">
        <label>
          <span>Samples</span>
          <textarea value={raw} onChange={(event) => setRaw(event.target.value)} rows={6} />
        </label>
        <button
          type="button"
          className="ghost"
          onClick={() => onApply(parsed)}
          disabled={parsed.length === 0}
        >
          Apply Dataset
        </button>
      </div>
    </div>
  );
}
