import { useEffect, useState, useCallback } from "react";
import { useDeepApi } from "../hooks/deep/useDeepApi";
import { useHotkeys } from "../hooks/common/useHotkeys";
import { DeepControls } from "../components/deep/DeepControls";
import { DeepStatsCard } from "../components/deep/DeepStatsCard";
import { DeepBoundaryCanvas } from "../components/deep/DeepBoundaryCanvas";
import { DeepComparisonTable } from "../components/deep/DeepComparisonTable";
import { DeepLossChart } from "../components/deep/DeepLossChart";
import { DeepEducation } from "../components/education/DeepEducation";
import "../styles/education.css";

const DATASETS = ["circles", "spiral", "xor", "baarle"];

export function DeepPage({ apiBase }: { apiBase: string }) {
  const {
    state,
    boundary,
    regions,
    history,
    comparison,
    error,
    loading,
    reset,
    step,
    trainEpoch,
    fetchBoundary,
    addToComparison,
    clearComparison,
  } = useDeepApi(apiBase);

  // Local state for controls
  const [localDataset, setLocalDataset] = useState("circles");
  const [localHiddenDims, setLocalHiddenDims] = useState([8, 8]);
  const [localLr, setLocalLr] = useState(0.1);
  const [localSeed, setLocalSeed] = useState(42);
  const [showTiling, setShowTiling] = useState(false);
  const [autoRefreshBoundary, setAutoRefreshBoundary] = useState(true);

  // Derived values
  const dataset = state?.dataset ?? localDataset;
  const architecture = state?.architecture ?? null;
  const metrics = state?.metrics ?? null;
  const samples = state?.samples ?? [];
  const epoch = state?.epoch ?? 0;
  const totalSteps = state?.total_steps ?? 0;

  // Initialize on mount
  useEffect(() => {
    void reset({
      dataset: localDataset,
      hidden_dims: localHiddenDims,
      lr: localLr,
      seed: localSeed,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refresh boundary when state changes (if auto-refresh enabled)
  useEffect(() => {
    if (state && autoRefreshBoundary) {
      void fetchBoundary(50);
    }
  }, [state?.total_steps, autoRefreshBoundary, fetchBoundary, state]);

  const handleReset = useCallback(() => {
    void reset({
      dataset: localDataset,
      hidden_dims: localHiddenDims,
      lr: localLr,
      seed: localSeed,
    });
  }, [reset, localDataset, localHiddenDims, localLr, localSeed]);

  const handleStep = useCallback(() => {
    void step(1);
  }, [step]);

  const handleTrainEpoch = useCallback(() => {
    void trainEpoch();
  }, [trainEpoch]);

  const handleTrain10Epochs = useCallback(async () => {
    for (let i = 0; i < 10; i++) {
      await trainEpoch();
    }
  }, [trainEpoch]);

  useHotkeys({
    onStep: handleStep,
    onReset: handleReset,
    enabled: Boolean(state),
  });

  const handleDatasetChange = (value: string) => {
    setLocalDataset(value);
    void reset({
      dataset: value,
      hidden_dims: localHiddenDims,
      lr: localLr,
      seed: localSeed,
    });
  };

  const handleHiddenDimsChange = (dims: number[]) => {
    setLocalHiddenDims(dims);
    void reset({
      dataset: localDataset,
      hidden_dims: dims,
      lr: localLr,
      seed: localSeed,
    });
  };

  const handleLrChange = (value: number) => {
    setLocalLr(value);
  };

  const handleSeedChange = (value: number) => {
    setLocalSeed(value);
    void reset({
      dataset: localDataset,
      hidden_dims: localHiddenDims,
      lr: localLr,
      seed: value,
    });
  };

  return (
    <section className="panel deep-panel">
      <header className="deep-header">
        <h2>Deep Learning Lab</h2>
        <p className="subtitle">Geometry of Depth — How layers fold the input space</p>
      </header>

      <div className="deep-grid">
        <DeepControls
          dataset={dataset}
          datasets={DATASETS}
          hiddenDims={localHiddenDims}
          lr={localLr}
          seed={localSeed}
          loading={loading}
          onDatasetChange={handleDatasetChange}
          onHiddenDimsChange={handleHiddenDimsChange}
          onLrChange={handleLrChange}
          onSeedChange={handleSeedChange}
        />

        <div className="deep-main">
          <div className="deep-actions">
            <button onClick={handleStep} disabled={loading}>
              Step (S)
            </button>
            <button onClick={handleTrainEpoch} disabled={loading}>
              Train Epoch
            </button>
            <button onClick={handleTrain10Epochs} disabled={loading}>
              Train ×10
            </button>
            <button onClick={handleReset} disabled={loading} className="reset-btn">
              Reset (R)
            </button>
          </div>

          <div className="view-toggle">
            <label>
              <input
                type="checkbox"
                checked={showTiling}
                onChange={(e) => setShowTiling(e.target.checked)}
              />
              Show Region Tiling
            </label>
            <label>
              <input
                type="checkbox"
                checked={autoRefreshBoundary}
                onChange={(e) => setAutoRefreshBoundary(e.target.checked)}
              />
              Auto-refresh visualization
            </label>
          </div>

          <DeepBoundaryCanvas
            boundary={boundary}
            samples={samples}
            showTiling={showTiling}
            dataset={dataset}
          />
        </div>

        <div className="deep-side">
          <DeepStatsCard
            architecture={architecture}
            metrics={metrics}
            regions={regions}
            epoch={epoch}
            totalSteps={totalSteps}
          />

          <DeepLossChart history={history} />

          <div className="comparison-actions">
            <button onClick={addToComparison} disabled={loading}>
              Add to Comparison
            </button>
          </div>
        </div>
      </div>

      <DeepComparisonTable entries={comparison} onClear={clearComparison} />

      <DeepEducation />

      {error && <p className="deep-error">{error}</p>}
    </section>
  );
}

