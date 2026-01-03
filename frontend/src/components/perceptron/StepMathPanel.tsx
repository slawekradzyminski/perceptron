import type { LastStep } from "../../types";
import { formatSign, formatVec } from "../../utils/visuals";

type StepMathPanelProps = {
  datasetName: string;
  w: number[];
  b: number;
  sampleIdx: number;
  nextInput: { x: number[]; y: number } | null;
  fallbackInput: number[];
  lastStep: LastStep | null;
  apiError: string | null;
};

export function StepMathPanel({
  datasetName,
  w,
  b,
  sampleIdx,
  nextInput,
  fallbackInput,
  lastStep,
  apiError,
}: StepMathPanelProps) {
  const nextX = nextInput?.x ?? fallbackInput;
  const nextY = nextInput?.y ?? -1;
  const biasInput = 1;
  const nextAugmented = [...nextX, biasInput];

  return (
    <section className="panel calc-panel step-math-panel">
      <h2>Step math</h2>
      <div className="calc-section">
        <h3>Current state</h3>
        <div className="calc-grid">
          <div><span>Weights w</span><strong>{formatVec(w)}</strong></div>
          <div><span>Bias b</span><strong>{b.toFixed(2)}</strong></div>
          <div><span>Dataset</span><strong>{datasetName}</strong></div>
          <div><span>Sample idx</span><strong>{sampleIdx}</strong></div>
        </div>
      </div>
      <div className="calc-section">
        <h3>Next input (what we expect)</h3>
        <div className="calc-grid">
          <div><span>Input x</span><strong>{formatVec(nextX)}</strong></div>
          <div><span>Label y</span><strong>{nextY === 1 ? "+1" : "-1"}</strong></div>
          <div><span>Bias input x_b</span><strong>{biasInput}</strong></div>
          <div><span>Augmented x̃</span><strong>{formatVec(nextAugmented)}</strong></div>
        </div>
      </div>
      {lastStep && (
        <div className="calc-section">
          <h3>Last step calculation</h3>
          <div className="calc-grid">
            <div><span>Input x</span><strong>{formatVec(lastStep.x)}</strong></div>
            <div><span>Label y</span><strong>{lastStep.y === 1 ? "+1" : "-1"}</strong></div>
            <div><span>Bias input x_b</span><strong>{biasInput}</strong></div>
            <div><span>Augmented x̃</span><strong>{formatVec([...lastStep.x, biasInput])}</strong></div>
            <div><span>Weights (after)</span><strong>{formatVec(w)}</strong></div>
            <div><span>Bias (after)</span><strong>{b.toFixed(2)}</strong></div>
            <div><span>Score s</span><strong>{lastStep.score.toFixed(2)}</strong></div>
            <div><span>Prediction ŷ</span><strong>{lastStep.pred === 1 ? "+1" : "-1"}</strong></div>
          </div>
          <p className="calc-eq">
            {(() => {
              const wBefore = w.map((val, i) => val - lastStep.deltaW[i]);
              const dot = wBefore.reduce((acc, val, i) => acc + val * lastStep.x[i], 0);
              return `s = w·x + b = ${dot.toFixed(2)} + ${formatSign(b - lastStep.deltaB)} = ${lastStep.score.toFixed(2)}`;
            })()}
          </p>
          <p className="calc-eq">
            {(() => {
              const wBefore = w.map((val, i) => val - lastStep.deltaW[i]);
              return `w_before + Δw = ${formatVec(wBefore)} + ${formatVec(lastStep.deltaW)} -> ${formatVec(w)},  b_before + Δb = ${(b - lastStep.deltaB).toFixed(2)} + ${lastStep.deltaB.toFixed(2)} -> ${b.toFixed(2)}`;
            })()}
          </p>
          <p className="calc-note">
            {lastStep.mistake
              ? `Mistake: y * s = ${(lastStep.y * lastStep.score).toFixed(2)} <= 0, update applied. Score uses pre-update weights.`
              : `Correct: y * s = ${(lastStep.y * lastStep.score).toFixed(2)} > 0, no update. Score uses pre-update weights.`}
          </p>
        </div>
      )}
      {!lastStep && apiError && (
        <div className="calc-section">
          <p className="calc-note">{apiError}</p>
        </div>
      )}
    </section>
  );
}
