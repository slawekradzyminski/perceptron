import { useEffect, useMemo, useRef } from "react";
import type { MlpTrainerSnapshot, Point } from "../types";
import { drawMlpBoundary } from "../utils/visuals";

type MlpBoundaryPanelProps = {
  snapshot: MlpTrainerSnapshot | null;
};

export function MlpBoundaryPanel({ snapshot }: MlpBoundaryPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const points: Point[] = useMemo(() => {
    if (!snapshot) return [];
    return snapshot.evals.map((row) => ({ x: row.x[0], y: row.x[1], label: row.y }));
  }, [snapshot]);
  const dim = snapshot ? snapshot.grid_rows * snapshot.grid_cols : 0;
  const showBoundary = Boolean(snapshot && dim === 2);

  useEffect(() => {
    if (!showBoundary || !snapshot) return;
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    drawMlpBoundary(
      ctx,
      points,
      snapshot.hidden.weights,
      snapshot.hidden.bias,
      snapshot.output.weights[0] ?? [],
      snapshot.output.bias[0] ?? 0,
    );
  }, [showBoundary, snapshot, points]);

  return (
    <section className="panel boundary-panel mlp-boundary">
      <h2>2D Boundary</h2>
      {!showBoundary && <p className="panel-subtle">Boundary is available only for 2D inputs.</p>}
      <canvas ref={canvasRef} className="plot-canvas" width={300} height={300} />
      <div className="panel-subtle mlp-boundary-legend">
        <span className="legend-line output">Output boundary</span>
        <span className="legend-line h1">Hidden h₁</span>
        <span className="legend-line h2">Hidden h₂</span>
      </div>
    </section>
  );
}
