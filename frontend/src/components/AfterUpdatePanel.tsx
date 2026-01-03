import { useEffect } from "react";
import type { Grid } from "../types";
import { drawGrid, setGridCanvasSize, valueColor } from "../utils/visuals";
import { getCellFromEvent } from "../utils/dom";

type AfterUpdatePanelProps = {
  gridRows: number;
  gridCols: number;
  weightsAfter: Grid;
  biasAfter: number;
  lastDeltaW: number[] | null;
  onTooltipChange: (next: { visible: boolean; text: string; x: number; y: number }) => void;
  onTooltipHide: () => void;
  weightsAfterRef: React.RefObject<HTMLCanvasElement>;
};

export function AfterUpdatePanel({
  gridRows,
  gridCols,
  weightsAfter,
  biasAfter,
  lastDeltaW,
  onTooltipChange,
  onTooltipHide,
  weightsAfterRef,
}: AfterUpdatePanelProps) {
  useEffect(() => {
    if (weightsAfterRef.current) setGridCanvasSize(weightsAfterRef.current, gridRows, gridCols);
  }, [gridRows, gridCols, weightsAfterRef]);

  useEffect(() => {
    if (weightsAfterRef.current) drawGrid(weightsAfterRef.current, weightsAfter, valueColor);
  }, [weightsAfter, weightsAfterRef]);

  const handleMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const { row, col } = getCellFromEvent(event, weightsAfter.length, weightsAfter[0].length);
    const v = weightsAfter[row][col];
    let detail = v.toFixed(2);
    if (lastDeltaW) {
      const index = row * gridCols + col;
      const prev = v - (lastDeltaW[index] ?? 0);
      detail = `${v.toFixed(2)} (prev ${prev.toFixed(2)}, Δ ${(lastDeltaW[index] ?? 0).toFixed(2)})`;
    }
    onTooltipChange({
      visible: true,
      text: `weights(after): ${detail}`,
      x: event.clientX + 12,
      y: event.clientY + 12,
    });
  };

  return (
    <section className="panel after-update-panel">
      <h2>After update</h2>
      <div className="switchboard">
        <div className="switch-row">
          <div>
            <h3>Weights (after)</h3>
            <canvas ref={weightsAfterRef} onMouseMove={handleMove} onMouseLeave={onTooltipHide} />
          </div>
          <div>
            <h3>Bias (after)</h3>
            <div
              className="bias-card bias-after"
              style={{ backgroundColor: valueColor(biasAfter), aspectRatio: `${gridCols} / ${gridRows}` }}
            >
              <div className="bias-value">{biasAfter.toFixed(2)}</div>
              <div className="bias-note">After update</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
