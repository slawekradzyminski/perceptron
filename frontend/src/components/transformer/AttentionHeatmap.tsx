import { useMemo, useState } from "react";

type AttentionHeatmapProps = {
  matrix: number[][];
  tokens: string[];
};

export function AttentionHeatmap({ matrix, tokens }: AttentionHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
  } | null>(null);

  // Calculate cell size based on token count (max ~400px canvas)
  const cellSize = useMemo(() => {
    const maxSize = 400;
    const n = tokens.length;
    return Math.max(12, Math.min(40, Math.floor(maxSize / n)));
  }, [tokens.length]);

  // Truncate token text for display
  const truncateToken = (text: string, maxLen: number = 8): string => {
    // Replace whitespace with visible indicator
    const display = text.replace(/^\s+/, "·").replace(/\s+$/, "·");
    if (display.length > maxLen) {
      return display.slice(0, maxLen - 1) + "…";
    }
    return display;
  };

  // Get color for attention weight (0-1)
  const getColor = (weight: number): string => {
    // Purple gradient: low = light, high = dark purple
    const intensity = Math.round(weight * 255);
    return `rgb(${255 - intensity * 0.3}, ${255 - intensity * 0.7}, ${255 - intensity * 0.1})`;
  };

  const hoveredWeight =
    hoveredCell !== null ? matrix[hoveredCell.row][hoveredCell.col] : null;

  return (
    <div className="heatmap-container">
      <div className="heatmap-wrapper">
        {/* Column labels (top) */}
        <div
          className="heatmap-col-labels"
          style={{ marginLeft: cellSize * 2 + 4 }}
        >
          {tokens.map((token, i) => (
            <div
              key={i}
              className={`heatmap-label col-label ${hoveredCell?.col === i ? "highlighted" : ""}`}
              style={{ width: cellSize }}
              title={token}
            >
              {truncateToken(token, 4)}
            </div>
          ))}
        </div>

        <div className="heatmap-main">
          {/* Row labels (left) */}
          <div className="heatmap-row-labels">
            {tokens.map((token, i) => (
              <div
                key={i}
                className={`heatmap-label row-label ${hoveredCell?.row === i ? "highlighted" : ""}`}
                style={{ height: cellSize, width: cellSize * 2 }}
                title={token}
              >
                {truncateToken(token, 6)}
              </div>
            ))}
          </div>

          {/* Heatmap grid */}
          <div
            className="heatmap-grid"
            style={{
              gridTemplateColumns: `repeat(${tokens.length}, ${cellSize}px)`,
              gridTemplateRows: `repeat(${tokens.length}, ${cellSize}px)`,
            }}
          >
            {matrix.map((row, rowIdx) =>
              row.map((weight, colIdx) => (
                <div
                  key={`${rowIdx}-${colIdx}`}
                  className={`heatmap-cell ${hoveredCell?.row === rowIdx || hoveredCell?.col === colIdx ? "cross-highlight" : ""}`}
                  style={{
                    backgroundColor: getColor(weight),
                    width: cellSize,
                    height: cellSize,
                  }}
                  onMouseEnter={() =>
                    setHoveredCell({ row: rowIdx, col: colIdx })
                  }
                  onMouseLeave={() => setHoveredCell(null)}
                  title={`${tokens[rowIdx]} → ${tokens[colIdx]}: ${weight.toFixed(3)}`}
                />
              )),
            )}
          </div>
        </div>
      </div>

      {/* Tooltip / info display */}
      {hoveredCell !== null && (
        <div className="heatmap-tooltip">
          <span className="token-from">"{tokens[hoveredCell.row]}"</span>
          <span className="arrow">→</span>
          <span className="token-to">"{tokens[hoveredCell.col]}"</span>
          <span className="weight">{hoveredWeight?.toFixed(4)}</span>
        </div>
      )}

      {/* Legend */}
      <div className="heatmap-legend">
        <span>Low attention</span>
        <div className="legend-gradient" />
        <span>High attention</span>
      </div>
    </div>
  );
}
