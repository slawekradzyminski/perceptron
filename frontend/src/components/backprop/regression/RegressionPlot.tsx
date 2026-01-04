import type { RegressionState, RegressionStep } from "../../types";

const WIDTH = 520;
const HEIGHT = 320;
const PAD = 36;

function extent(values: number[]) {
  return [Math.min(...values), Math.max(...values)] as const;
}

function scale(value: number, min: number, max: number, span: number) {
  if (max === min) return 0.5 * span;
  return ((value - min) / (max - min)) * span;
}

type RegressionPlotProps = {
  state: RegressionState | null;
  history: RegressionStep[];
};

export function RegressionPlot({ state, history }: RegressionPlotProps) {
  const samples = state?.samples ?? [];
  if (!samples.length) {
    return (
      <div className="panel backprop-card">
        <h3>Regression Plot</h3>
        <p className="diag-placeholder">Regression samples are not available yet.</p>
      </div>
    );
  }

  const xs = samples.map((s) => s.x);
  const ys = samples.map((s) => s.y);
  const [minX, maxX] = extent(xs);
  const [minY, maxY] = extent(ys);
  const xSpan = WIDTH - 2 * PAD;
  const ySpan = HEIGHT - 2 * PAD;

  const toX = (x: number) => PAD + scale(x, minX, maxX, xSpan);
  const toY = (y: number) => HEIGHT - PAD - scale(y, minY, maxY, ySpan);

  const linePoints = (m: number, b: number) => {
    const y1 = m * minX + b;
    const y2 = m * maxX + b;
    return `${toX(minX)},${toY(y1)} ${toX(maxX)},${toY(y2)}`;
  };

  const step0 = history[0]?.params_before ?? state?.params;
  const step1 = history[0]?.params_after;
  const step7 = history[6]?.params_after;

  return (
    <div className="panel backprop-card">
      <h3>Regression Plot</h3>
      <svg width={WIDTH} height={HEIGHT} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Regression plot">
        <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="#fffdf8" stroke="#d1bda7" />
        <line x1={PAD} y1={HEIGHT - PAD} x2={WIDTH - PAD} y2={HEIGHT - PAD} stroke="#1c1b19" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={HEIGHT - PAD} stroke="#1c1b19" />

        {samples.map((s, idx) => (
          <circle key={idx} cx={toX(s.x)} cy={toY(s.y)} r={4} fill="#ff6a3d" />
        ))}

        {step0 && (
          <polyline points={linePoints(step0.m, step0.b)} fill="none" stroke="#2a7b8f" strokeDasharray="4 3" strokeWidth={2} />
        )}
        {step1 && (
          <polyline points={linePoints(step1.m, step1.b)} fill="none" stroke="#2f7d4f" strokeDasharray="6 4" strokeWidth={2} />
        )}
        {step7 && (
          <polyline points={linePoints(step7.m, step7.b)} fill="none" stroke="#1c1b19" strokeWidth={2} />
        )}
      </svg>
      <div className="backprop-legend">
        <span className="legend-line h1">step 0</span>
        <span className="legend-line h2">step 1</span>
        <span className="legend-line">step 7</span>
      </div>
    </div>
  );
}
