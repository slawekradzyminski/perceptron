import type { ChangeEvent } from "react";

const DATASET_OPTIONS = [
  { value: "or", label: "OR" },
  { value: "xor", label: "XOR" },
  { value: "custom", label: "Custom" },
];

type AppRoute = "main" | "lms" | "mlp" | "gd" | "tinygps" | "regression" | "deep" | "alexnet" | "transformer" | "convolution";

type HeaderProps = {
  datasetName: string;
  gridRows: number;
  gridCols: number;
  lr: number;
  showCustomButton: boolean;
  route: AppRoute;
  showTrainingControls: boolean;
  onDatasetChange: (value: string) => void;
  onStep: () => void;
  onReset: () => void;
  onLrChange: (value: number) => void;
  onOpenCustom: () => void;
  onRouteChange: (route: AppRoute) => void;
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
  onLrChange,
  onOpenCustom,
  onRouteChange,
}: HeaderProps) {
  const handleDataset = (event: ChangeEvent<HTMLSelectElement>) => {
    onDatasetChange(event.target.value);
  };

  return (
    <header className="hero">
      <div className="hero-title">
        <h1>AI Learning Lab</h1>
        <p className="keyboard-hint">Press <kbd>S</kbd> for step, <kbd>R</kbd> for reset</p>
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
            Perceptron
          </button>
          <button
            type="button"
            className={route === "lms" ? "active" : ""}
            onClick={() => onRouteChange("lms")}
            role="tab"
            aria-selected={route === "lms"}
          >
            LMS
          </button>
          <button
            type="button"
            className={route === "mlp" ? "active" : ""}
            onClick={() => onRouteChange("mlp")}
            role="tab"
            aria-selected={route === "mlp"}
          >
            MLP
          </button>
          <button
            type="button"
            className={route === "gd" ? "active" : ""}
            onClick={() => onRouteChange("gd")}
            role="tab"
            aria-selected={route === "gd"}
          >
            GD
          </button>
          <button
            type="button"
            className={route === "tinygps" ? "active" : ""}
            onClick={() => onRouteChange("tinygps")}
            role="tab"
            aria-selected={route === "tinygps"}
          >
            TinyGPS
          </button>
          <button
            type="button"
            className={route === "regression" ? "active" : ""}
            onClick={() => onRouteChange("regression")}
            role="tab"
            aria-selected={route === "regression"}
          >
            Regression
          </button>
          <button
            type="button"
            className={route === "deep" ? "active" : ""}
            onClick={() => onRouteChange("deep")}
            role="tab"
            aria-selected={route === "deep"}
          >
            Deep
          </button>
          <button
            type="button"
            className={route === "alexnet" ? "active" : ""}
            onClick={() => onRouteChange("alexnet")}
            role="tab"
            aria-selected={route === "alexnet"}
          >
            AlexNet
          </button>
          <button
            type="button"
            className={route === "transformer" ? "active" : ""}
            onClick={() => onRouteChange("transformer")}
            role="tab"
            aria-selected={route === "transformer"}
          >
            Transformer
          </button>
          <button
            type="button"
            className={route === "convolution" ? "active" : ""}
            onClick={() => onRouteChange("convolution")}
            role="tab"
            aria-selected={route === "convolution"}
          >
            Conv Lab
          </button>
        </div>
        {route !== "gd" && route !== "tinygps" && route !== "regression" && route !== "deep" && route !== "alexnet" && route !== "transformer" && route !== "convolution" && (
          <>
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
          </>
        )}
        {showCustomButton && route !== "tinygps" && route !== "regression" && (
          <button type="button" className="ghost" onClick={onOpenCustom}>
            Customize
          </button>
        )}
        {showTrainingControls && (
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
        )}
      </div>
    </header>
  );
}
