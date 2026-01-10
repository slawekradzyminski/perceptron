type ReceptiveFieldBounds = {
  startRow: number;
  startCol: number;
  endRow: number;
  endCol: number;
} | null;

type ConvImageGridProps = {
  image: number[][];
  imageSize: number;
  imageName: string;
  inputShape: [number, number];
  receptiveField: ReceptiveFieldBounds;
};

export function ConvImageGrid({
  image,
  imageSize,
  imageName,
  inputShape,
  receptiveField,
}: ConvImageGridProps) {
  return (
    <div className="viz-panel">
      <h3>Input Image</h3>
      <div className="image-grid-container">
        <div
          className="image-grid"
          style={{
            gridTemplateColumns: `repeat(${imageSize}, 1fr)`,
          }}
        >
          {image.flatMap((row, r) =>
            row.map((val, c) => {
              const isInReceptiveField =
                receptiveField &&
                r >= receptiveField.startRow &&
                r < receptiveField.endRow &&
                c >= receptiveField.startCol &&
                c < receptiveField.endCol;
              const brightness = Math.round(val);
              return (
                <div
                  key={`${r}-${c}`}
                  className={`pixel-cell ${isInReceptiveField ? "highlighted" : ""}`}
                  style={{
                    backgroundColor: `rgb(${brightness}, ${brightness}, ${brightness})`,
                  }}
                  title={`(${r}, ${c}): ${val.toFixed(0)}`}
                />
              );
            }),
          )}
        </div>
      </div>
      <p className="viz-label">
        {imageName} ({inputShape[0]}×{inputShape[1]})
      </p>
    </div>
  );
}
