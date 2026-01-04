import { useEffect, useState } from "react";
import { useBackpropApi } from "../hooks/backprop/useBackpropApi";
import { BackpropHeader } from "../components/backprop/common/BackpropHeader";
import { RegressionControls } from "../components/backprop/regression/RegressionControls";
import { RegressionStepCard } from "../components/backprop/regression/RegressionStepCard";
import { RegressionHistoryTable } from "../components/backprop/regression/RegressionHistoryTable";
import { RegressionPlot } from "../components/backprop/regression/RegressionPlot";
import { RegressionDatasetEditor } from "../components/backprop/regression/RegressionDatasetEditor";
import { RegressionBookTable } from "../components/backprop/regression/RegressionBookTable";
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
    loading,
    resetRegression,
    stepRegression,
  } = useBackpropApi(apiBase);
  const [loss, setLoss] = useState<"mse" | "l1">("mse");
  const [lr, setLr] = useState(DEFAULT_LR);
  const [bookMode, setBookMode] = useState(false);
  const [bookLiteral, setBookLiteral] = useState(true);

  useEffect(() => {
    if (!state) return;
    setLoss(state.regression.loss);
    setLr(state.regression.lr);
  }, [state?.regression.loss, state?.regression.lr]);

  const step = lastEntry<RegressionStep>(regressionHistory);

  const handleStep = () => {
    if (!state) return;
    void stepRegression();
  };

  const handleReset = () => {
    if (!state) return;
    void resetRegression(loss, lr);
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
    setLoss(nextLoss);
    setBookMode(true);
    setBookLiteral(true);
    await resetRegression(nextLoss, 0.1, bookSamples, bookOrder, { m: 1, b: 0 });
  };

  const runSteps = async (count: number) => {
    for (let i = 0; i < count; i += 1) {
      await stepRegression();
    }
  };

  return (
    <section className="panel regression-panel">
      <BackpropHeader
        loading={loading}
        hasState={Boolean(state)}
        onStepPrimary={handleStep}
        primaryLabel="Step Regression"
      />

      <div className="regression-grid">
        <RegressionControls
          loss={loss}
          lr={lr}
          loading={loading}
          hasState={Boolean(state)}
          onLossChange={(value) => {
            setLoss(value);
            if (state) void resetRegression(value, lr);
          }}
          onLrChange={(value) => setLr(value)}
          onLrCommit={() => {
            if (state) void resetRegression(loss, lr);
          }}
          onReset={handleReset}
        />

        <RegressionStepCard state={state?.regression ?? null} step={step} />
        <RegressionPlot state={state?.regression ?? null} history={regressionHistory} />
        <RegressionDatasetEditor
          samples={state?.regression.samples ?? []}
          onApply={(samples) => resetRegression(loss, lr, samples)}
        />
        <div className="panel backprop-card">
          <h3>Book Presets</h3>
          <p className="panel-subtle">Load the exact regression dataset/order and initial parameters.</p>
          <div className="backprop-controls">
            <button type="button" onClick={() => void applyBookPreset("mse")}>
              Load MSE (Book)
            </button>
            <button type="button" onClick={() => void applyBookPreset("l1")}>
              Load L1 (Book)
            </button>
          </div>
        </div>
      </div>

      <div className="regression-bottom">
        <RegressionHistoryTable history={regressionHistory} />
        <div className="panel backprop-card">
          <h3>Book Table Mode</h3>
          <div className="backprop-controls">
            <label>
              <span>Show Book Table</span>
              <select value={bookMode ? "on" : "off"} onChange={(event) => setBookMode(event.target.value === "on")}>
                <option value="off">Off</option>
                <option value="on">On</option>
              </select>
            </label>
            <label>
              <span>Loss display</span>
              <select
                value={bookLiteral ? "book" : "computed"}
                onChange={(event) => setBookLiteral(event.target.value === "book")}
              >
                <option value="book">Book-literal</option>
                <option value="computed">Computed</option>
              </select>
            </label>
            <button type="button" onClick={() => void runSteps(8)} disabled={!state}>
              Run 8 Steps
            </button>
          </div>
        </div>
        {bookMode && <RegressionBookTable history={regressionHistory} lossType={loss} bookLiteral={bookLiteral} />}
      </div>

      {error && <p className="diag-error">{error}</p>}
    </section>
  );
}
