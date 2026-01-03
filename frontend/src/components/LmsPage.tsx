import { useEffect, useMemo, useRef, useState } from "react";
import { useLmsApi } from "../hooks/useLmsApi";
import { useHotkeys } from "../hooks/useHotkeys";
import type { CustomConfig } from "../types";
import { BoundaryPanel } from "./BoundaryPanel";
import { pointsForDataset } from "../utils/custom";
import { LmsDatasetCard } from "./LmsDatasetCard";
import { LmsMathCard } from "./LmsMathCard";
import { LmsTable } from "./LmsTable";
import { LmsTooltip } from "./LmsTooltip";
import { LmsTrendCard } from "./LmsTrendCard";

type LmsPageProps = {
  apiBase: string;
  datasetName: string;
  customConfig: CustomConfig;
  customApplied: boolean;
};

const OR_DATASET = [
  { x: [-1, -1], y: -1 },
  { x: [-1, 1], y: 1 },
  { x: [1, -1], y: 1 },
  { x: [1, 1], y: 1 },
];

const XOR_DATASET = [
  { x: [-1, -1], y: -1 },
  { x: [-1, 1], y: 1 },
  { x: [1, -1], y: 1 },
  { x: [1, 1], y: -1 },
];

function datasetForLms(datasetName: string, customConfig: CustomConfig) {
  if (datasetName === "xor") return XOR_DATASET;
  if (datasetName === "custom") {
    return customConfig.samples
      .map((sample) => {
        const flat = sample.grid.flat();
        if (flat.length !== 2) return null;
        return { x: [flat[0], flat[1]], y: sample.y };
      })
      .filter((row): row is { x: number[]; y: number } => row !== null);
  }
  return OR_DATASET;
}

function formatVec(values: number[]) {
  return `[${values.map((val) => val.toFixed(2)).join(", ")}]`;
}

export function LmsPage({ apiBase, datasetName, customConfig, customApplied }: LmsPageProps) {
  const { state, history, stepCount, error, loading, step, reset, resetWithOptions } = useLmsApi(apiBase);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ visible: boolean; text: string; x: number; y: number }>({
    visible: false,
    text: "",
    x: 0,
    y: 0,
  });
  const [lmsLr, setLmsLr] = useState(0.1);
  const datasetRows = useMemo(() => datasetForLms(datasetName, customConfig), [datasetName, customConfig]);
  const datasetDim = datasetName === "custom" ? customConfig.rows * customConfig.cols : 2;
  const canUseCustom = datasetName !== "custom" || (customApplied && datasetDim === 2);
  const points = useMemo(
    () =>
      pointsForDataset(datasetName, datasetDim, customConfig),
    [datasetName, datasetDim, customConfig],
  );
  const w = state?.w ?? [0, 0];
  const b = state?.b ?? 0;
  const firstStepNumber = stepCount > history.length ? stepCount - history.length + 1 : 1;

  useEffect(() => {
    if (!canUseCustom) {
      return;
    }
    void resetWithOptions({ datasetName, customConfig, customApplied, lr: lmsLr });
  }, [resetWithOptions, datasetName, customConfig, customApplied, canUseCustom, lmsLr]);

  useHotkeys({ onStep: step, onReset: reset, enabled: canUseCustom });

  return (
    <section className="panel lms-panel">
      <div className="lms-head">
        <div>
          <h2>LMS (Widrow–Hoff) Exercise</h2>
          <p className="panel-subtle">
            We minimize squared error by nudging weights in the direction of the gradient.
          </p>
        </div>
        <div className="lms-actions">
          <button type="button" onClick={step} disabled={loading || !canUseCustom}>
            Step
          </button>
          <button type="button" onClick={reset} disabled={loading || !canUseCustom}>
            Reset
          </button>
        </div>
      </div>

      <div className="lms-grid">
        <BoundaryPanel w={w} b={b} points={points} show={true} canvasRef={canvasRef} />
        <LmsMathCard />
        <LmsDatasetCard
          datasetName={datasetName}
          datasetDim={datasetDim}
          datasetRows={datasetRows}
          canUseCustom={canUseCustom}
          lmsLr={lmsLr}
          onLrChange={setLmsLr}
          state={state}
          formatVec={formatVec}
        />
        <LmsTrendCard history={history} stepCount={stepCount} />
      </div>

      {error && <p className="diag-error">{error}</p>}

      <div className="lms-table-wrap">
        <LmsTable
          history={history}
          stepCount={stepCount}
          firstStepNumber={firstStepNumber}
          onHover={(payload) => setTooltip(payload)}
          onLeave={() => setTooltip((prev) => ({ ...prev, visible: false }))}
        />
      </div>
      <LmsTooltip tooltip={tooltip} />
    </section>
  );
}
