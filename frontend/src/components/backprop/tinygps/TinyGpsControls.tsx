type TinyGpsControlsProps = {
  datasets: string[];
  dataset: string;
  lr: number;
  coords: Array<[string, [number, number][]]>;
  onDatasetChange: (value: string) => void;
  onLrChange: (value: number) => void;
  onLrCommit?: () => void;
};

export function TinyGpsControls({
  datasets,
  dataset,
  lr,
  coords,
  onDatasetChange,
  onLrChange,
  onLrCommit,
}: TinyGpsControlsProps) {
  return (
    <div className="panel backprop-card">
      <h3>TinyGPS Controls</h3>
      <div className="backprop-controls">
        <label>
          <span>Dataset</span>
          <select value={dataset} onChange={(event) => onDatasetChange(event.target.value)}>
            {datasets.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Learning rate</span>
          <div className="lr-row">
            <input
              type="range"
              min="0.01"
              max="1"
              step="0.01"
              value={lr}
              onChange={(event) => onLrChange(Number(event.target.value))}
              onMouseUp={() => onLrCommit?.()}
            />
            <span>{lr.toFixed(2)}</span>
          </div>
        </label>
        {coords.length > 0 && (
          <div className="tinygps-coords-inline">
            {coords.map(([city, points]) => (
              <div key={city} className="tinygps-coords-block">
                <strong>{city}</strong>
                <ul>
                  {points.map(([lat, lon], idx) => (
                    <li key={`${city}-${idx}`}>
                      ({lat.toFixed(4)}, {lon.toFixed(4)})
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
