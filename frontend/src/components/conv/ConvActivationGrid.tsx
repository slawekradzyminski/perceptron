type ConvActivationGridProps = {
  activationMap: number[][];
  activationMapNormalized: number[][];
  outputShape: [number, number];
  hoveredCell: { row: number; col: number } | null;
  onCellHover: (cell: { row: number; col: number } | null) => void;
};

export function ConvActivationGrid({
  activationMap,
  activationMapNormalized,
  outputShape,
  hoveredCell,
  onCellHover,
}: ConvActivationGridProps) {
  if (outputShape[0] <= 0) {
    return null;
  }

  return (
    <div className="viz-panel">
      <h3>Activation Map</h3>
      <div className="activation-grid-container">
        <div
          className="activation-grid"
          style={{
            gridTemplateColumns: `repeat(${outputShape[1]}, 1fr)`,
          }}
        >
          {activationMapNormalized.flatMap((row, r) =>
            row.map((val, c) => {
              const isHovered =
                hoveredCell?.row === r && hoveredCell?.col === c;
              const brightness = Math.round(val);
              return (
                <div
                  key={`${r}-${c}`}
                  className={`activation-cell ${isHovered ? "hovered" : ""}`}
                  style={{
                    backgroundColor: `rgb(${brightness}, ${brightness}, ${brightness})`,
                  }}
                  onMouseEnter={() => onCellHover({ row: r, col: c })}
                  onMouseLeave={() => onCellHover(null)}
                  title={`(${r}, ${c}): ${activationMap[r][c].toFixed(2)}`}
                />
              );
            }),
          )}
        </div>
      </div>
      <p className="viz-label">
        output ({outputShape[0]}×{outputShape[1]})
      </p>
      <p className="viz-hint">Hover over cells to see calculation</p>
    </div>
  );
}
