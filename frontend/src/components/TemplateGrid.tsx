import { useEffect, useRef } from "react";
import { drawGrid, setGridCanvasSize, valueColor } from "../utils/visuals";

export function TemplateGrid({ label, grid }: { label: string; grid: number[][] }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    setGridCanvasSize(ref.current, grid.length, grid[0].length);
    drawGrid(ref.current, grid, valueColor);
  }, [grid]);

  return (
    <div className="template-card">
      <span>{label}</span>
      <canvas ref={ref} aria-label={label} />
    </div>
  );
}
