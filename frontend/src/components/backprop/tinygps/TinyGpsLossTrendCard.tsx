import type { TinyGpsStep } from "../../types";

type TinyGpsLossTrendCardProps = {
  history: TinyGpsStep[];
};

export function TinyGpsLossTrendCard({ history }: TinyGpsLossTrendCardProps) {
  if (!history.length) {
    return (
      <div className="panel backprop-card tinygps-loss-card">
        <h3>Loss Trend</h3>
        <p className="diag-placeholder">Step TinyGPS to plot loss over time.</p>
      </div>
    );
  }

  const padding = 16;
  const width = 420;
  const height = 240;
  const innerW = width - padding * 2;
  const innerH = height - padding * 2 - 6;
  const losses = history.map((step) => step.loss);
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const range = max - min || 1;

  const points = losses.map((loss, idx) => {
    const x = padding + (idx / Math.max(1, losses.length - 1)) * innerW;
    const y = padding + innerH - ((loss - min) / range) * innerH;
    return { x, y };
  });

  const path = points
    .map((pt, idx) => `${idx === 0 ? "M" : "L"}${pt.x.toFixed(2)},${pt.y.toFixed(2)}`)
    .join(" ");

  return (
    <div className="panel backprop-card tinygps-loss-card">
      <h3>Loss Trend</h3>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Loss trend" className="tinygps-loss-svg">
        <rect x={0} y={0} width={width} height={height} rx={12} fill="#fffdf8" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding - 6} stroke="#cbb8a4" />
        <line
          x1={padding}
          y1={height - padding - 6}
          x2={width - padding}
          y2={height - padding - 6}
          stroke="#cbb8a4"
        />
        <line
          x1={padding}
          y1={padding}
          x2={width - padding}
          y2={padding}
          stroke="#f0e4d6"
          strokeDasharray="4 4"
        />
        <text x={padding + 2} y={padding - 4} fontSize={10} fill="#7b6a57">
          {max.toFixed(3)}
        </text>
        <text x={padding + 2} y={height - padding + 10} fontSize={10} fill="#7b6a57">
          {min.toFixed(3)}
        </text>
        <text x={padding + 2} y={padding + 12} fontSize={9} fill="#9c8975">
          higher loss ↑
        </text>
        <path d={path} fill="none" stroke="#e26b3c" strokeWidth={2.5} />
        {points.map((pt, idx) => (
          <circle key={idx} cx={pt.x} cy={pt.y} r={3.2} fill="#e26b3c" />
        ))}
      </svg>
      <div className="diag-note">step → (loss can rise or fall)</div>
    </div>
  );
}
