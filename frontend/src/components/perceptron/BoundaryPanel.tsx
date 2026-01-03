import { useEffect } from "react";
import type { Point } from "../../types";
import { drawBoundary, drawPoints } from "../../utils/visuals";

type BoundaryPanelProps = {
  w: number[];
  b: number;
  points: Point[];
  show: boolean;
  canvasRef: React.RefObject<HTMLCanvasElement>;
};

export function BoundaryPanel({ w, b, points, show, canvasRef }: BoundaryPanelProps) {
  useEffect(() => {
    if (!show) return;
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    drawBoundary(ctx, w, b);
    drawPoints(ctx, points);
  }, [show, w, b, points, canvasRef]);

  if (!show) return null;

  return (
    <section className="panel boundary-panel" id="boundary-panel">
      <h2>2D Boundary</h2>
      <canvas ref={canvasRef} id="plot" className="plot-canvas" width={420} height={420} />
    </section>
  );
}
