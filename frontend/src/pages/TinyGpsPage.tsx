import { useMemo, useState } from "react";
import { useBackpropApi } from "../hooks/backprop/useBackpropApi";
import { BackpropHeader } from "../components/backprop/common/BackpropHeader";
import { TinyGpsControls } from "../components/backprop/tinygps/TinyGpsControls";
import { TinyGpsBookTable } from "../components/backprop/tinygps/TinyGpsBookTable";
import { TinyGpsBackpropDiagram } from "../components/backprop/tinygps/TinyGpsBackpropDiagram";
import { TinyGpsCityMap } from "../components/backprop/tinygps/TinyGpsCityMap";
import { TinyGpsLossTrendCard } from "../components/backprop/tinygps/TinyGpsLossTrendCard";
import { TinyGpsStepSetupCard } from "../components/backprop/tinygps/TinyGpsStepSetupCard";
import { useHotkeys } from "../hooks/common/useHotkeys";

const DEFAULT_LR = 0.1;

export function TinyGpsPage({ apiBase }: { apiBase: string }) {
  const {
    state,
    tinygpsHistory,
    error,
    loading: _loading,
    resetTinygps,
    stepTinygps,
  } = useBackpropApi(apiBase);
  // Local state for user selections (controls UI before reset is called)
  const [localDataset, setLocalDataset] = useState("madrid-paris-berlin");
  const [localLr, setLocalLr] = useState(DEFAULT_LR);
  const step = tinygpsHistory.length ? tinygpsHistory[tinygpsHistory.length - 1] : null;

  // Use backend state if available, otherwise fall back to local
  const dataset = state?.tinygps.dataset ?? localDataset;
  const lr = state?.tinygps.lr ?? localLr;

  const coordsList = useMemo(() => {
    if (!state) return [];
    const byCity = new Map<string, [number, number][]>();
    for (const sample of state.tinygps.samples) {
      const list = byCity.get(sample.city) ?? [];
      list.push([sample.lat, sample.lon]);
      byCity.set(sample.city, list);
    }
    return Array.from(byCity.entries());
  }, [state]);

  const handleStep = () => {
    if (!state) return;
    void stepTinygps();
  };

  const handleReset = () => {
    if (!state) return;
    void resetTinygps(localDataset, localLr);
  };

  useHotkeys({ onStep: handleStep, onReset: handleReset, enabled: Boolean(state) });

  return (
    <section className="panel tinygps-panel">
      <BackpropHeader />

      <div className="tinygps-grid">
        <TinyGpsControls
          datasets={state?.tinygps_datasets ?? []}
          dataset={dataset}
          lr={lr}
          coords={coordsList}
          onDatasetChange={(value) => {
            setLocalDataset(value);
            if (state) void resetTinygps(value, localLr);
          }}
          onLrChange={(value) => setLocalLr(value)}
          onLrCommit={() => {
            if (state) void resetTinygps(localDataset, localLr);
          }}
        />

        <TinyGpsStepSetupCard
          state={state?.tinygps ?? null}
          onApply={(params, order) => resetTinygps(dataset, lr, params, order)}
        />

        <TinyGpsLossTrendCard history={tinygpsHistory} />
      </div>

      <div className="tinygps-grid tinygps-grid-secondary">
        <TinyGpsBackpropDiagram step={step} />
        <TinyGpsCityMap state={state?.tinygps ?? null} />
      </div>

      <div className="tinygps-history">
        <TinyGpsBookTable history={tinygpsHistory} datasetName={dataset} />
      </div>

      {error && <p className="diag-error">{error}</p>}
    </section>
  );
}
