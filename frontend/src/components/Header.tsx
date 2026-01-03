import type { ChangeEvent } from "react";

const DATASET_OPTIONS = [
  { value: "or", label: "OR" },
  { value: "xor", label: "XOR" },
  { value: "custom", label: "Custom" },
];

type HeaderProps = {
  datasetName: string;
  gridRows: number;
  gridCols: number;
  lr: number;
  showCustomButton: boolean;
  route: "main" | "diagnostics";
  showTrainingControls: boolean;
  onDatasetChange: (value: string) => void;
  onStep: () => void;
  onReset: () => void;
  onLrChange: (value: number) => void;
  onOpenCustom: () => void;
  onRouteChange: (route: "main" | "diagnostics") => void;
};

export function Header({
  datasetName,
  gridRows,
  gridCols,
  lr,
  showCustomButton,
  route,
  showTrainingControls,
  onDatasetChange,
  onStep,
  onReset,
  onLrChange,
  onOpenCustom,
  onRouteChange,
}: HeaderProps) {
  const handleDataset = (event: ChangeEvent<HTMLSelectElement>) => {
    onDatasetChange(event.target.value);
  };

  return (
    <header className="hero">
      <div>
        <p className="eyebrow">Perceptron Visual Lab</p>
        <h1>Make the decision boundary move.</h1>
        <p className="subhead">A hands-on view of weights, mistakes, and linearly separable data.</p>
      </div>
      <div className="controls">
        <div className="view-toggle" role="tablist" aria-label="View">
          <button
            type="button"
            className={route === "main" ? "active" : ""}
            onClick={() => onRouteChange("main")}
            role="tab"
            aria-selected={route === "main"}
          >
            Main
          </button>
          <button
            type="button"
            className={route === "diagnostics" ? "active" : ""}
            onClick={() => onRouteChange("diagnostics")}
            role="tab"
            aria-selected={route === "diagnostics"}
          >
            Diagnostics
          </button>
        </div>
        <label>
          <span>Dataset</span>
          <select value={datasetName} onChange={handleDataset} aria-label="Dataset">
            {DATASET_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <div className="static-field">
          <span>Switchboard</span>
          <strong>{gridRows}×{gridCols}</strong>
        </div>
        {showCustomButton && (
          <button type="button" className="ghost" onClick={onOpenCustom}>
            Customize
          </button>
        )}
        {showTrainingControls && (
          <>
            <button onClick={onStep}>Step</button>
            <button onClick={onReset}>Reset</button>
            <label>
              <span>Learning rate</span>
              <div className="lr-row">
                <input
                  type="range"
                  min="0.1"
                  max="2"
                  step="0.1"
                  value={lr}
                  onChange={(event) => onLrChange(Number(event.target.value))}
                />
                <span>{lr.toFixed(1)}</span>
              </div>
            </label>
          </>
        )}
        {showTrainingControls && (
          <p className="hotkeys">Shortcuts: <strong>S</strong> step, <strong>R</strong> reset</p>
        )}
      </div>
    </header>
  );
}
