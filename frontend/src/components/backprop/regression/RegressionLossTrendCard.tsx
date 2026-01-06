import type { RegressionStep } from "../../../types";

type RegressionLossTrendCardProps = {
  history: RegressionStep[];
  lossType: "mse" | "l1";
};

export function RegressionLossTrendCard({ history, lossType }: RegressionLossTrendCardProps) {
  if (!history.length) {
    return (
      <div className="panel backprop-card regression-loss-card">
        <h3>Loss Trend ({lossType.toUpperCase()})</h3>
        <p className="diag-placeholder">Step through regression to plot loss over time.</p>
      </div>
    );
  }

  const padding = 20;
  const width = 400;
  const height = 200;
  const innerW = width - padding * 2;
  const innerH = height - padding * 2 - 6;
  const losses = history.map((step) => step.loss_value);
  const min = Math.min(...losses, 0);
  const max = Math.max(...losses);
  const range = max - min || 1;

  const points = losses.map((loss, idx) => {
    const x = padding + (idx / Math.max(1, losses.length - 1)) * innerW;
    const y = padding + innerH - ((loss - min) / range) * innerH;
    return { x, y, loss };
  });

  const path = points
    .map((pt, idx) => `${idx === 0 ? "M" : "L"}${pt.x.toFixed(2)},${pt.y.toFixed(2)}`)
    .join(" ");

  const color = lossType === "mse" ? "#3d5af1" : "#8b5cf6";

  return (
    <div className="panel backprop-card regression-loss-card">
      <h3>Loss Trend ({lossType.toUpperCase()})</h3>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Loss trend" className="regression-loss-svg">
        <rect x={0} y={0} width={width} height={height} rx={12} fill="#fffdf8" />
        
        {/* Axes */}
        <line x1={padding} y1={padding} x2={padding} y2={height - padding - 6} stroke="#cbb8a4" />
        <line x1={padding} y1={height - padding - 6} x2={width - padding} y2={height - padding - 6} stroke="#cbb8a4" />
        
        {/* Grid lines */}
        <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="#f0e4d6" strokeDasharray="4 4" />
        <line x1={padding} y1={(padding + height - padding - 6) / 2} x2={width - padding} y2={(padding + height - padding - 6) / 2} stroke="#f0e4d6" strokeDasharray="4 4" />
        
        {/* Labels */}
        <text x={padding + 2} y={padding - 4} fontSize={10} fill="#7b6a57">
          {max.toFixed(2)}
        </text>
        <text x={padding + 2} y={height - padding + 10} fontSize={10} fill="#7b6a57">
          {min.toFixed(2)}
        </text>
        <text x={width - padding} y={height - padding + 10} fontSize={10} fill="#7b6a57" textAnchor="end">
          step {history.length - 1}
        </text>

        {/* Area under curve */}
        <path
          d={`${path} L${points[points.length - 1]?.x ?? padding},${height - padding - 6} L${padding},${height - padding - 6} Z`}
          fill={color}
          fillOpacity={0.1}
        />
        
        {/* Line */}
        <path d={path} fill="none" stroke={color} strokeWidth={2.5} />
        
        {/* Points */}
        {points.map((pt, idx) => (
          <g key={idx}>
            <circle cx={pt.x} cy={pt.y} r={4} fill={color} />
            <title>Step {idx}: Loss = {pt.loss.toFixed(4)}</title>
          </g>
        ))}
      </svg>
      <div className="diag-note">
        Loss {losses[losses.length - 1] < losses[0] ? "decreasing ↓" : "fluctuating"} — 
        latest: {losses[losses.length - 1]?.toFixed(4)}
      </div>
    </div>
  );
}

