type LossPoint = { p: number; l1: number; ce: number };

function makePath(points: LossPoint[], width: number, height: number, valueKey: "l1" | "ce") {
  const maxValue = Math.max(...points.map((point) => point[valueKey]));
  const minP = Math.min(...points.map((point) => point.p));
  const maxP = Math.max(...points.map((point) => point.p));
  const scaleX = (p: number) => (p - minP) / (maxP - minP) * width;
  const scaleY = (v: number) => height - (v / maxValue) * height;
  return points
    .map((point, index) => {
      const x = scaleX(point.p);
      const y = scaleY(point[valueKey]);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

type LossComparisonCardProps = {
  points: LossPoint[];
};

export function LossComparisonCard({ points }: LossComparisonCardProps) {
  const width = 280;
  const height = 140;
  const l1Path = makePath(points, width, height, "l1");
  const cePath = makePath(points, width, height, "ce");

  return (
    <div className="lms-math lms-loss-compare">
      <h3>Loss penalty vs p(correct)</h3>
      <p>Cross-entropy grows much faster when the model is confidently wrong.</p>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Loss curves">
        <path className="lms-loss-l1" d={l1Path} />
        <path className="lms-loss-ce" d={cePath} />
      </svg>
      <div className="lms-loss-legend">
        <span className="lms-loss-dot l1" />
        <span>L1 loss</span>
        <span className="lms-loss-dot ce" />
        <span>Cross-entropy</span>
      </div>
    </div>
  );
}
