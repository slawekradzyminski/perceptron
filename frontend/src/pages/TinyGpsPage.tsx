import { useEffect, useMemo, useState } from "react";
import { useBackpropApi } from "../hooks/backprop/useBackpropApi";
import { BackpropHeader } from "../components/backprop/common/BackpropHeader";
import { TinyGpsControls } from "../components/backprop/tinygps/TinyGpsControls";
import { TinyGpsMapCard } from "../components/backprop/tinygps/TinyGpsMapCard";
import { TinyGpsBookTable } from "../components/backprop/tinygps/TinyGpsBookTable";
import { TinyGpsStepSetupCard } from "../components/backprop/tinygps/TinyGpsStepSetupCard";
import { useHotkeys } from "../hooks/common/useHotkeys";

const DEFAULT_LR = 0.1;

export function TinyGpsPage({ apiBase }: { apiBase: string }) {
  const {
    state,
    tinygpsHistory,
    error,
    loading,
    resetTinygps,
    stepTinygps,
  } = useBackpropApi(apiBase);
  const [dataset, setDataset] = useState("madrid-paris-berlin");
  const [lr, setLr] = useState(DEFAULT_LR);

  useEffect(() => {
    if (!state) return;
    setDataset(state.tinygps.dataset);
    setLr(state.tinygps.lr);
  }, [state?.tinygps.dataset, state?.tinygps.lr]);

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
    void resetTinygps(dataset, lr);
  };

  useHotkeys({ onStep: handleStep, onReset: handleReset, enabled: Boolean(state) });

  return (
    <section className="panel tinygps-panel">
      <BackpropHeader
        loading={loading}
        hasState={Boolean(state)}
        onStepPrimary={() => stepTinygps()}
        primaryLabel="Step TinyGPS"
      />

      <div className="tinygps-grid">
        <TinyGpsControls
          datasets={state?.tinygps_datasets ?? []}
          dataset={dataset}
          lr={lr}
          loading={loading}
          hasState={Boolean(state)}
          coords={coordsList}
          onDatasetChange={(value) => {
            setDataset(value);
            if (state) void resetTinygps(value, lr);
          }}
          onLrChange={(value) => setLr(value)}
          onLrCommit={() => {
            if (state) void resetTinygps(dataset, lr);
          }}
          onReset={handleReset}
        />

        <TinyGpsStepSetupCard
          state={state?.tinygps ?? null}
          onApply={(params, order) => resetTinygps(dataset, lr, params, order)}
        />

        <TinyGpsMapCard />
      </div>

      <div className="tinygps-history">
        <TinyGpsBookTable history={tinygpsHistory} datasetName={dataset} />
      </div>

      {error && <p className="diag-error">{error}</p>}
    </section>
  );
}
