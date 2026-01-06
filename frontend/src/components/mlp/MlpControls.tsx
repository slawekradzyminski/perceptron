type MlpControlsProps = {
  hiddenDim: number;
  lr: number;
  seed: number;
  loading: boolean;
  disabled: boolean;
  onHiddenDimChange: (value: number) => void;
  onLrChange: (value: number) => void;
  onSeedChange: (value: number) => void;
};

export function MlpControls({
  hiddenDim,
  lr,
  seed,
  onHiddenDimChange,
  onLrChange,
  onSeedChange,
}: MlpControlsProps) {
  return (
    <div className="mlp-controls">
      <h3>MLP Configuration</h3>
      <div className="diag-controls compact">
        <label>
          Hidden units
          <input
            type="number"
            min={1}
            max={8}
            value={hiddenDim}
            onChange={(event) => onHiddenDimChange(Number(event.target.value))}
          />
        </label>
        <label>
          Learning rate
          <input
            type="number"
            step="0.05"
            min={0.05}
            value={lr}
            onChange={(event) => onLrChange(Number(event.target.value))}
          />
        </label>
        <label>
          Seed
          <input type="number" value={seed} onChange={(event) => onSeedChange(Number(event.target.value))} />
        </label>
      </div>
    </div>
  );
}
