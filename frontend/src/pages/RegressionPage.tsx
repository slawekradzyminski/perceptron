import { useState } from "react";
import { useBackpropApi } from "../hooks/backprop/useBackpropApi";
import { BackpropHeader } from "../components/backprop/common/BackpropHeader";
import { RegressionControls } from "../components/backprop/regression/RegressionControls";
import { RegressionPlot } from "../components/backprop/regression/RegressionPlot";
import { RegressionBookTable } from "../components/backprop/regression/RegressionBookTable";
import { RegressionLossTrendCard } from "../components/backprop/regression/RegressionLossTrendCard";
import { useHotkeys } from "../hooks/common/useHotkeys";
import type { RegressionStep } from "../types";

const DEFAULT_LR = 0.1;

function lastEntry<T>(rows: T[]) {
  return rows.length ? rows[rows.length - 1] : null;
}

export function RegressionPage({ apiBase }: { apiBase: string }) {
  const {
    state,
    regressionHistory,
    error,
    loading: _loading,
    resetRegression,
    stepRegression,
  } = useBackpropApi(apiBase);
  const [bookLiteral, setBookLiteral] = useState(true);
  // Local state for user selections (controls UI before reset is called)
  const [localLoss, setLocalLoss] = useState<"mse" | "l1">("mse");
  const [localLr, setLocalLr] = useState(DEFAULT_LR);

  // Use backend state if available, otherwise fall back to local
  const loss = state?.regression.loss ?? localLoss;
  const lr = state?.regression.lr ?? localLr;

  const step = lastEntry<RegressionStep>(regressionHistory);

  const handleStep = () => {
    if (!state) return;
    void stepRegression();
  };

  const handleReset = () => {
    if (!state) return;
    void resetRegression(localLoss, localLr);
  };

  useHotkeys({ onStep: handleStep, onReset: handleReset, enabled: Boolean(state) });

  const bookSamples = [
    { x: 1, y: 3 },
    { x: 2, y: 5 },
    { x: 3, y: 7 },
    { x: 4, y: 9 },
  ];
  const bookOrder = [0, 1, 2, 3, 0, 1, 2, 3];

  const applyBookPreset = async (nextLoss: "mse" | "l1") => {
    setLocalLoss(nextLoss);
    setBookLiteral(true);
    await resetRegression(nextLoss, 0.1, bookSamples, bookOrder, { m: 1, b: 0 });
  };

  const samples = state?.regression.samples ?? [];
  const currentParams = step?.params_after ?? state?.regression.params ?? { m: 0, b: 0 };

  return (
    <section className="panel regression-panel">
      <BackpropHeader />

      {/* Top row: Controls, Setup, Loss Trend */}
      <div className="regression-grid">
        <RegressionControls
          loss={loss}
          lr={lr}
          onLossChange={(value) => {
            setLocalLoss(value);
            if (state) void resetRegression(value, localLr);
          }}
          onLrChange={(value) => setLocalLr(value)}
          onLrCommit={() => {
            if (state) void resetRegression(localLoss, localLr);
          }}
        />

        {/* Exercise Setup Card */}
        <div className="panel backprop-card">
          <h3>Book Exercise Setup</h3>
          <p className="diag-note" style={{ marginBottom: 12 }}>
            Load the exact dataset and initial parameters from exercises 3.18-3.29.
          </p>

          {/* Current formula */}
          <div className="regression-formula">
            <div className="regression-formula-row">
              <span className="regression-formula-label">Model:</span>
              <span className="regression-formula-value">
                ŷ = {currentParams.m.toFixed(3)} × x + {currentParams.b.toFixed(3)}
              </span>
            </div>
            <div className="regression-formula-row">
              <span className="regression-formula-label">Loss ({loss.toUpperCase()}):</span>
              <span className="regression-formula-value">
                {loss === "mse" ? "L = (ŷ - y)²" : "L = |ŷ - y|"}
              </span>
            </div>
          </div>

          {/* Dataset display */}
          <div className="regression-dataset">
            {samples.slice(0, 4).map((s, idx) => (
              <div key={idx} className="regression-dataset-point">
                <span className="x">x={s.x}</span>
                <span className="y"> y={s.y}</span>
              </div>
            ))}
          </div>

          {/* Preset buttons */}
          <div className="regression-presets" style={{ marginTop: 12 }}>
            <button type="button" onClick={() => void applyBookPreset("mse")}>
              Load MSE (Ex. 3.18)
            </button>
            <button type="button" onClick={() => void applyBookPreset("l1")}>
              Load L1 (Ex. 3.24)
            </button>
          </div>
        </div>

        {/* Loss Trend */}
        <RegressionLossTrendCard history={regressionHistory} lossType={loss} />
      </div>

      {/* Secondary row: Plot and Book Table */}
      <div className="regression-grid-secondary">
        <RegressionPlot state={state?.regression ?? null} history={regressionHistory} />
        <RegressionBookTable history={regressionHistory} lossType={loss} bookLiteral={bookLiteral} />
      </div>

      {error && <p className="diag-error">{error}</p>}
    </section>
  );
}
