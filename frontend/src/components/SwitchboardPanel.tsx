import { useEffect } from "react";
import type { Grid, TooltipState } from "../types";
import { drawGrid, setGridCanvasSize, valueColor } from "../utils/visuals";
import { getCellFromEvent } from "../utils/dom";
import { Tooltip } from "./Tooltip";

type SwitchboardPanelProps = {
  gridRows: number;
  gridCols: number;
  inputGrid: Grid;
  weightsBefore: Grid;
  contribGrid: Grid;
  biasBefore: number;
  displayScore: number;
  inputRef: React.RefObject<HTMLCanvasElement>;
  weightsBeforeRef: React.RefObject<HTMLCanvasElement>;
  contribRef: React.RefObject<HTMLCanvasElement>;
  tooltip: TooltipState;
  onTooltipChange: (next: TooltipState) => void;
  onTooltipHide: () => void;
};

export function SwitchboardPanel({
  gridRows,
  gridCols,
  inputGrid,
  weightsBefore,
  contribGrid,
  biasBefore,
  displayScore,
  inputRef,
  weightsBeforeRef,
  contribRef,
  tooltip,
  onTooltipChange,
  onTooltipHide,
}: SwitchboardPanelProps) {
  useEffect(() => {
    if (inputRef.current) setGridCanvasSize(inputRef.current, gridRows, gridCols);
    if (weightsBeforeRef.current) setGridCanvasSize(weightsBeforeRef.current, gridRows, gridCols);
    if (contribRef.current) setGridCanvasSize(contribRef.current, gridRows, gridCols);
  }, [gridRows, gridCols, inputRef, weightsBeforeRef, contribRef]);

  useEffect(() => {
    if (inputRef.current) drawGrid(inputRef.current, inputGrid, valueColor);
    if (weightsBeforeRef.current) drawGrid(weightsBeforeRef.current, weightsBefore, valueColor);
    if (contribRef.current) drawGrid(contribRef.current, contribGrid, valueColor);
  }, [inputGrid, weightsBefore, contribGrid, inputRef, weightsBeforeRef, contribRef]);

  const handleMove = (
    grid: Grid,
    label: string,
    formatter: (row: number, col: number) => string,
  ) => (event: React.MouseEvent<HTMLCanvasElement>) => {
    const { row, col } = getCellFromEvent(event, grid.length, grid[0].length);
    onTooltipChange({
      visible: true,
      text: `${label}: ${formatter(row, col)}`,
      x: event.clientX + 12,
      y: event.clientY + 12,
    });
  };

  return (
    <section className="panel switchboard-panel">
      <h2>Switchboard</h2>
      <div className="switchboard">
        <div className="switch-row">
          <div>
            <h3>Input</h3>
            <canvas
              ref={inputRef}
              onMouseMove={handleMove(inputGrid, "input", (row, col) => {
                const v = inputGrid[row][col];
                return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
              })}
              onMouseLeave={onTooltipHide}
            />
          </div>
          <div>
            <h3>Weights (before)</h3>
            <canvas
              ref={weightsBeforeRef}
              onMouseMove={handleMove(weightsBefore, "weights(before)", (row, col) => {
                const v = weightsBefore[row][col];
                return v.toFixed(2);
              })}
              onMouseLeave={onTooltipHide}
            />
          </div>
        </div>
        <div className="switch-row">
          <div>
            <h3>Contribution (pre-update)</h3>
            <canvas
              ref={contribRef}
              onMouseMove={handleMove(contribGrid, "contrib(pre)", (row, col) => {
                const v = contribGrid[row][col];
                return `${v.toFixed(2)} = x${row + 1}${col + 1}*w${row + 1}${col + 1}`;
              })}
              onMouseLeave={onTooltipHide}
            />
          </div>
          <div>
            <h3>Bias (before)</h3>
            <div
              className="bias-card"
              style={{ backgroundColor: valueColor(biasBefore), aspectRatio: `${gridCols} / ${gridRows}` }}
            >
              <div className="bias-value">{biasBefore.toFixed(2)}</div>
              <div className="bias-note">Adds to score</div>
            </div>
          </div>
        </div>
      </div>
      <Tooltip tooltip={tooltip} />
      <div className="meter">
        <span>Score (s)</span>
        <strong data-testid="score-value">{displayScore.toFixed(2)}</strong>
      </div>
    </section>
  );
}
