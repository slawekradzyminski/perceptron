import type { LmsStep } from "../types";

function ErrorTrend({ history, stepCount }: { history: LmsStep[]; stepCount: number }) {
  if (history.length === 0) {
    return <p className="lms-empty">Step to see the error curve.</p>;
  }
  const points = history.map((row) => row.error * row.error);
  const maxVal = Math.max(...points, 1e-6);
  const minVal = Math.min(...points);
  const height = 90;
  const width = 260;
  const padding = 8;
  const span = Math.max(maxVal - minVal, 1e-6);
  const step = points.length > 1 ? width / (points.length - 1) : width;
  const path = points
    .map((val, idx) => {
      const x = idx * step;
      const y = height - padding - ((val - minVal) / span) * (height - padding * 2);
      return `${idx === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const latest = history[history.length - 1];
  const latestE = latest.error * latest.error;
  return (
    <div className="lms-trend">
      <div className="lms-trend-header">
        <div>
          <strong>Latest step:</strong> {stepCount}
        </div>
        <div className="lms-trend-meta">
          <span>Window: {history.length}/32</span>
          <span>E={latestE.toFixed(3)}</span>
          <span>ŷ={latest.y_hat.toFixed(2)}</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Error trend">
        <defs>
          <linearGradient id="lmsTrend" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#d36a56" />
            <stop offset="100%" stopColor="#b94a3b" />
          </linearGradient>
        </defs>
        <path d={path} />
        {points.map((val, idx) => {
          const x = idx * step;
          const y = height - padding - ((val - minVal) / span) * (height - padding * 2);
          return <circle key={idx} cx={x} cy={y} r={2.2} />;
        })}
      </svg>
      <div className="lms-trend-range">E range (last {history.length} steps)</div>
      <div className="lms-trend-axis">
        <span>Min E: {minVal.toFixed(2)}</span>
        <span>Max E: {maxVal.toFixed(2)}</span>
      </div>
    </div>
  );
}

export function LmsTrendCard({ history, stepCount }: { history: LmsStep[]; stepCount: number }) {
  return (
    <div className="lms-math">
      <h3>Error trend (E = (y − ŷ)²)</h3>
      <ErrorTrend history={history} stepCount={stepCount} />
    </div>
  );
}
