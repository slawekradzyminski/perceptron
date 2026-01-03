import { useEffect, useMemo, useState } from "react";
import { useMlpTrainerApi } from "../hooks/mlp/useMlpTrainerApi";
import { TemplateGrid } from "../components/mlp/TemplateGrid";
import { useHotkeys } from "../hooks/common/useHotkeys";
import { MlpBoundaryPanel } from "../components/mlp/MlpBoundaryPanel";
import { MlpControls } from "../components/mlp/MlpControls";
import { MlpSummaryCard } from "../components/mlp/MlpSummaryCard";
import { MlpMathCard } from "../components/mlp/MlpMathCard";
import { MlpWeightList } from "../components/mlp/MlpWeightList";
import { MlpNetworkDiagram } from "../components/mlp/MlpNetworkDiagram";
import { MlpEvalList } from "../components/mlp/MlpEvalList";
import { MlpStepTemplates } from "../components/mlp/MlpStepTemplates";
import type { CustomConfig } from "../types";

export type MlpTrainerPageProps = {
  apiBase: string;
  datasetName: string;
  customConfig: CustomConfig;
  customApplied: boolean;
};

export function MlpTrainerPage({ apiBase, datasetName, customConfig, customApplied }: MlpTrainerPageProps) {
  const { snapshot, lastStep, error, loading, resetWithOptions, step } = useMlpTrainerApi(apiBase);
  const [hiddenDim, setHiddenDim] = useState(2);
  const [mlpLr, setMlpLr] = useState(0.5);
  const [seed, setSeed] = useState(0);

  const canUseCustom = datasetName !== "custom" || customApplied;

  useEffect(() => {
    if (!canUseCustom) return;
    void resetWithOptions({
      datasetName,
      customConfig,
      customApplied,
      hiddenDim,
      lr: mlpLr,
      seed,
    });
  }, [canUseCustom, datasetName, customConfig, customApplied, hiddenDim, mlpLr, seed, resetWithOptions]);

  const handleStep = () => {
    if (!canUseCustom) return;
    void step();
  };
  const handleReset = () => {
    if (!canUseCustom) return;
    void resetWithOptions({ datasetName, customConfig, customApplied, hiddenDim, lr: mlpLr, seed });
  };

  useHotkeys({ onStep: handleStep, onReset: handleReset, enabled: canUseCustom });

  const evalSummary = useMemo(() => {
    if (!snapshot) return { correct: 0, total: 0 };
    const correct = snapshot.evals.filter((row) => row.pred === row.y).length;
    return { correct, total: snapshot.evals.length };
  }, [snapshot]);

  return (
    <section className="panel mlp-train">
      <div className="mlp-grid">
        <MlpControls
          hiddenDim={hiddenDim}
          lr={mlpLr}
          seed={seed}
          loading={loading}
          disabled={!canUseCustom}
          onHiddenDimChange={setHiddenDim}
          onLrChange={setMlpLr}
          onSeedChange={setSeed}
          onStep={handleStep}
          onReset={handleReset}
        />

        <MlpBoundaryPanel snapshot={snapshot} />
        {snapshot && <MlpNetworkDiagram snapshot={snapshot} />}
        <div className="mlp-side">
          {snapshot && (
            <MlpSummaryCard snapshot={snapshot} correct={evalSummary.correct} total={evalSummary.total} />
          )}
          <MlpMathCard />
          {snapshot && <MlpWeightList snapshot={snapshot} />}
        </div>

        <div className="mlp-templates">
          {snapshot && (
            <div className="template-section">
              <h3>Current hidden templates</h3>
              <div className="template-grid">
                {snapshot.hidden.templates.map((grid, idx) => (
                  <TemplateGrid key={`current-${idx}`} label={`H${idx + 1}`} grid={grid} />
                ))}
              </div>
            </div>
          )}
          {lastStep && <MlpStepTemplates step={lastStep} />}
        </div>

        <div className="mlp-preds">
          {snapshot && <MlpEvalList evals={snapshot.evals} />}
        </div>
      </div>

      {!canUseCustom && <p className="diag-note">Apply the custom dataset to train the MLP.</p>}
      {error && <p className="diag-error">{error}</p>}
    </section>
  );
}
