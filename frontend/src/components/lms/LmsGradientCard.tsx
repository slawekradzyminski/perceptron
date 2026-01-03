import type { LmsStep } from "../../types";

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function LmsGradientCard({ history }: { history: LmsStep[] }) {
  if (history.length === 0) {
    return (
      <div className="lms-math">
        <h3>Gradient view</h3>
        <p className="lms-empty">Step to see the gradient vector.</p>
      </div>
    );
  }

  const latest = history[history.length - 1];
  const wBefore = latest.w_before;
  const grad = [latest.grad_w1, latest.grad_w2];
  const delta = [latest.w_after[0] - wBefore[0], latest.w_after[1] - wBefore[1]];

  const width = 220;
  const height = 220;
  const padding = 18;
  const min = -2;
  const max = 2;
  const span = max - min;

  const mapX = (val: number) => padding + ((val - min) / span) * (width - padding * 2);
  const mapY = (val: number) => height - padding - ((val - min) / span) * (height - padding * 2);

  const origin = {
    x: mapX(clamp(wBefore[0], min, max)),
    y: mapY(clamp(wBefore[1], min, max)),
  };

  const scale = 0.4;
  const gradEnd = {
    x: mapX(clamp(wBefore[0] + grad[0] * scale, min, max)),
    y: mapY(clamp(wBefore[1] + grad[1] * scale, min, max)),
  };
  const updateEnd = {
    x: mapX(clamp(wBefore[0] + delta[0] * 4, min, max)),
    y: mapY(clamp(wBefore[1] + delta[1] * 4, min, max)),
  };

  return (
    <div className="lms-math">
      <h3>Gradient view</h3>
      <p className="lms-note">Vectors are drawn in weight space (w₁, w₂).</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="lms-grad" role="img" aria-label="Gradient plot">
        <rect x={padding} y={padding} width={width - padding * 2} height={height - padding * 2} rx={12} />
        <line x1={mapX(0)} y1={padding} x2={mapX(0)} y2={height - padding} />
        <line x1={padding} y1={mapY(0)} x2={width - padding} y2={mapY(0)} />
        <circle cx={origin.x} cy={origin.y} r={4} />
        <line x1={origin.x} y1={origin.y} x2={gradEnd.x} y2={gradEnd.y} className="grad" />
        <line x1={origin.x} y1={origin.y} x2={updateEnd.x} y2={updateEnd.y} className="update" />
      </svg>
      <div className="lms-grad-legend">
        <span>∇E = [{grad[0].toFixed(2)}, {grad[1].toFixed(2)}]</span>
        <span>Δw = [{delta[0].toFixed(2)}, {delta[1].toFixed(2)}]</span>
      </div>
    </div>
  );
}
