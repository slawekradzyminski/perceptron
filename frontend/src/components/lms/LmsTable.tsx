import type { LmsStep } from "../../types";

type TooltipPayload = { visible: boolean; text: string; x: number; y: number };

type LmsRowProps = {
  step: LmsStep;
  index: number;
  firstStepNumber: number;
  onHover: (payload: TooltipPayload) => void;
  onLeave: () => void;
};

function LmsRow({ step, index, firstStepNumber, onHover, onLeave }: LmsRowProps) {
  const correct = step.y * step.y_hat > 0;
  const displayStep = firstStepNumber + index;
  const eta = step.lr;
  const [x1, x2] = step.x;
  const w1 = step.w_before[0];
  const w2 = step.w_before[1];
  const s = step.y_hat;
  const e = step.error;
  const tooltip = [
    `Step ${displayStep}`,
    `ŷ = w1*x1 + w2*x2 + b`,
    `= (${w1.toFixed(2)}*${x1.toFixed(0)}) + (${w2.toFixed(2)}*${x2.toFixed(0)}) + ${step.b_before.toFixed(2)}`,
    `= ${s.toFixed(2)}`,
    `e = y - ŷ = ${step.y.toFixed(0)} - ${s.toFixed(2)} = ${e.toFixed(2)}`,
    `Δw1 = η·e·x1 = ${eta.toFixed(2)}·${e.toFixed(2)}·${x1.toFixed(0)} = ${(eta * e * x1).toFixed(2)}`,
    `Δw2 = η·e·x2 = ${eta.toFixed(2)}·${e.toFixed(2)}·${x2.toFixed(0)} = ${(eta * e * x2).toFixed(2)}`,
    `Δb = η·e = ${eta.toFixed(2)}·${e.toFixed(2)} = ${(eta * e).toFixed(2)}`,
    `E = e² = ${(e * e).toFixed(3)}`,
  ].join("\n");
  return (
    <tr
      onMouseMove={(event) => {
        const padding = 12;
        const boxWidth = 320;
        const boxHeight = 190;
        const viewportW = window.innerWidth;
        const viewportH = window.innerHeight;
        const flipX = event.clientX + padding + boxWidth > viewportW;
        const flipY = event.clientY + padding + boxHeight > viewportH;
        const x = Math.max(
          8,
          Math.min(viewportW - boxWidth - 8, event.clientX + (flipX ? -boxWidth - padding : padding)),
        );
        const y = Math.max(
          8,
          Math.min(viewportH - boxHeight - 8, event.clientY + (flipY ? -boxHeight - padding : padding)),
        );
        onHover({
          visible: true,
          text: tooltip,
          x,
          y,
        });
      }}
      onMouseLeave={onLeave}
    >
      <td>{displayStep}</td>
      <td>{step.x[0].toFixed(0)}</td>
      <td>{step.x[1].toFixed(0)}</td>
      <td>1</td>
      <td>{step.y.toFixed(0)}</td>
      <td>{step.w_before[0].toFixed(2)}</td>
      <td>{step.w_before[1].toFixed(2)}</td>
      <td>{step.b_before.toFixed(2)}</td>
      <td>{step.y_hat.toFixed(2)}</td>
      <td className={correct ? "ok" : "bad"}>{correct ? "✓" : "✗"}</td>
      <td>{step.w_after[0].toFixed(2)}</td>
      <td>{step.w_after[1].toFixed(2)}</td>
      <td>{step.b_after.toFixed(2)}</td>
      <td>{step.grad_w1.toFixed(2)}</td>
      <td>{step.grad_w2.toFixed(2)}</td>
      <td>{step.grad_b.toFixed(2)}</td>
    </tr>
  );
}

export function LmsTable({
  history,
  stepCount,
  firstStepNumber,
  onHover,
  onLeave,
}: {
  history: LmsStep[];
  stepCount: number;
  firstStepNumber: number;
  onHover: (payload: TooltipPayload) => void;
  onLeave: () => void;
}) {
  return (
    <table className="lms-table">
      <thead>
        <tr>
          <th>Step</th>
          <th>x₁</th>
          <th>x₂</th>
          <th>x_b</th>
          <th>y</th>
          <th>w₁</th>
          <th>w₂</th>
          <th>b</th>
          <th>ŷ</th>
          <th>✓?</th>
          <th>w₁′</th>
          <th>w₂′</th>
          <th>b′</th>
          <th>∂E/∂w₁</th>
          <th>∂E/∂w₂</th>
          <th>∂E/∂b</th>
        </tr>
      </thead>
      <tbody>
        {history.length === 0 && (
          <tr>
            <td colSpan={16} className="lms-empty">
              Click Step to start filling the LMS table.
            </td>
          </tr>
        )}
        {history.map((row, index) => (
          <LmsRow
            key={`${row.idx}-${index}`}
            step={row}
            index={index}
            firstStepNumber={firstStepNumber}
            onHover={onHover}
            onLeave={onLeave}
          />
        ))}
      </tbody>
    </table>
  );
}
