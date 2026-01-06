import type { RegressionState, RegressionStep } from "../../../types";

const WIDTH = 450;
const HEIGHT = 280;
const PAD = 40;

function _extent(values: number[]) {
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
  const padX = 0.5;
  const padY = 1;
  const [minX, maxX] = [Math.min(...xs) - padX, Math.max(...xs) + padX];
  const [minY, maxY] = [Math.min(...ys) - padY, Math.max(...ys) + padY];
  const xSpan = WIDTH - 2 * PAD;
  const ySpan = HEIGHT - 2 * PAD;

  const toX = (x: number) => PAD + scale(x, minX, maxX, xSpan);
  const toY = (y: number) => HEIGHT - PAD - scale(y, minY, maxY, ySpan);

  const linePoints = (m: number, b: number) => {
    const y1 = m * minX + b;
    const y2 = m * maxX + b;
    return `${toX(minX)},${toY(y1)} ${toX(maxX)},${toY(y2)}`;
  };

  // Get initial and current parameters
  const initialParams = history[0]?.params_before ?? state?.params;
  const currentParams = history.length > 0 
    ? history[history.length - 1]?.params_after 
    : state?.params;

  // Grid lines
  const xTicks = [1, 2, 3, 4];
  const yTicks = [2, 4, 6, 8];

  const stepCount = history.length;

  return (
    <div className="panel backprop-card">
      <h3>Fitted Line (Step {stepCount})</h3>
      <svg 
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`} 
        role="img" 
        aria-label="Regression plot"
        style={{ width: "100%", maxWidth: 500, height: "auto", display: "block" }}
      >
        {/* Background */}
        <rect x={0} y={0} width={WIDTH} height={HEIGHT} rx={12} fill="#fffdf8" />
        
        {/* Grid */}
        {xTicks.map((x) => (
          <line key={`gx-${x}`} x1={toX(x)} y1={PAD} x2={toX(x)} y2={HEIGHT - PAD} stroke="#f0e4d6" strokeDasharray="4 4" />
        ))}
        {yTicks.map((y) => (
          <line key={`gy-${y}`} x1={PAD} y1={toY(y)} x2={WIDTH - PAD} y2={toY(y)} stroke="#f0e4d6" strokeDasharray="4 4" />
        ))}

        {/* Axes */}
        <line x1={PAD} y1={HEIGHT - PAD} x2={WIDTH - PAD} y2={HEIGHT - PAD} stroke="#1c1b19" strokeWidth={2} />
        <line x1={PAD} y1={PAD} x2={PAD} y2={HEIGHT - PAD} stroke="#1c1b19" strokeWidth={2} />

        {/* Axis labels */}
        {xTicks.map((x) => (
          <text key={`lx-${x}`} x={toX(x)} y={HEIGHT - PAD + 16} fontSize={11} textAnchor="middle" fill="#6a5a4a">
            {x}
          </text>
        ))}
        {yTicks.map((y) => (
          <text key={`ly-${y}`} x={PAD - 6} y={toY(y) + 4} fontSize={11} textAnchor="end" fill="#6a5a4a">
            {y}
          </text>
        ))}
        <text x={WIDTH / 2} y={HEIGHT - 6} fontSize={11} textAnchor="middle" fill="#6a5a4a" fontWeight={600}>
          x
        </text>
        <text x={12} y={HEIGHT / 2} fontSize={11} textAnchor="middle" fill="#6a5a4a" fontWeight={600} transform={`rotate(-90, 12, ${HEIGHT / 2})`}>
          y
        </text>

        {/* Initial line (faded) */}
        {initialParams && stepCount > 0 && (
          <polyline 
            points={linePoints(initialParams.m, initialParams.b)} 
            fill="none" 
            stroke="#cbb8a4" 
            strokeDasharray="6 4" 
            strokeWidth={2} 
          />
        )}

        {/* Current fitted line */}
        {currentParams && (
          <polyline 
            points={linePoints(currentParams.m, currentParams.b)} 
            fill="none" 
            stroke="#3d5af1" 
            strokeWidth={3} 
          />
        )}

        {/* Data points */}
        {samples.map((s, idx) => (
          <g key={idx}>
            <circle cx={toX(s.x)} cy={toY(s.y)} r={7} fill="#e26b3c" stroke="#fff" strokeWidth={2} />
            <title>({s.x}, {s.y})</title>
          </g>
        ))}
      </svg>

      {/* Legend */}
      <div className="regression-legend">
        {stepCount > 0 && (
          <div className="regression-legend-item">
            <span className="regression-legend-line step0" />
            <span>Initial (m={initialParams?.m.toFixed(1)}, b={initialParams?.b.toFixed(1)})</span>
          </div>
        )}
        <div className="regression-legend-item">
          <span className="regression-legend-line" style={{ background: "#3d5af1" }} />
          <span>Current (m={currentParams?.m.toFixed(3)}, b={currentParams?.b.toFixed(3)})</span>
        </div>
      </div>

      <p className="diag-note">
        Target: y = 2x + 1. 
        {currentParams && ` Error: Δm=${Math.abs(currentParams.m - 2).toFixed(3)}, Δb=${Math.abs(currentParams.b - 1).toFixed(3)}`}
      </p>
    </div>
  );
}
