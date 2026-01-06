import { DeepNetworkPreview } from "./DeepNetworkPreview";

type DeepControlsProps = {
  dataset: string;
  datasets: string[];
  hiddenDims: number[];
  lr: number;
  seed: number;
  loading: boolean;
  onDatasetChange: (value: string) => void;
  onHiddenDimsChange: (value: number[]) => void;
  onLrChange: (value: number) => void;
  onSeedChange: (value: number) => void;
};

export function DeepControls({
  dataset,
  datasets,
  hiddenDims,
  lr,
  seed,
  loading,
  onDatasetChange,
  onHiddenDimsChange,
  onLrChange,
  onSeedChange,
}: DeepControlsProps) {
  const depth = hiddenDims.length;
  const width = hiddenDims[0] || 8;

  const handleDepthChange = (newDepth: number) => {
    const newDims = Array.from({ length: newDepth }, () => width);
    onHiddenDimsChange(newDims);
  };

  const handleWidthChange = (newWidth: number) => {
    const newDims = Array.from({ length: depth }, () => newWidth);
    onHiddenDimsChange(newDims);
  };

  return (
    <div className="deep-controls">
      <h3>Architecture</h3>

      <label>
        Dataset
        <select
          value={dataset}
          onChange={(e) => onDatasetChange(e.target.value)}
          disabled={loading}
        >
          {datasets.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </label>

      <label>
        Depth (layers)
        <input
          type="range"
          min={1}
          max={6}
          value={depth}
          onChange={(e) => handleDepthChange(Number(e.target.value))}
          disabled={loading}
        />
        <span className="value-display">{depth}</span>
      </label>

      <label>
        Width (neurons)
        <input
          type="range"
          min={4}
          max={64}
          step={4}
          value={width}
          onChange={(e) => handleWidthChange(Number(e.target.value))}
          disabled={loading}
        />
        <span className="value-display">{width}</span>
      </label>

      <label>
        Learning Rate
        <input
          type="number"
          min={0.001}
          max={1}
          step={0.01}
          value={lr}
          onChange={(e) => onLrChange(Number(e.target.value))}
          disabled={loading}
        />
      </label>

      <label>
        Seed
        <input
          type="number"
          min={0}
          max={9999}
          value={seed}
          onChange={(e) => onSeedChange(Number(e.target.value))}
          disabled={loading}
        />
      </label>

      <DeepNetworkPreview
        inputDim={2}
        hiddenDims={hiddenDims}
        outputDim={2}
      />
    </div>
  );
}

