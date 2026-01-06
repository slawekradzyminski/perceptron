import type { DeepComparisonEntry } from "../../hooks/deep/useDeepApi";

type DeepComparisonTableProps = {
  entries: DeepComparisonEntry[];
  onClear: () => void;
};

export function DeepComparisonTable({ entries, onClear }: DeepComparisonTableProps) {
  if (entries.length === 0) {
    return (
      <div className="deep-comparison empty">
        <h3>Architecture Comparison (Exercise 4.9)</h3>
        <p className="hint">
          Train different architectures and click "Add to Comparison" to build a table
          comparing regions across depth/width configurations.
        </p>
      </div>
    );
  }

  return (
    <div className="deep-comparison">
      <div className="comparison-header">
        <h3>Architecture Comparison</h3>
        <button onClick={onClear} className="clear-btn">
          Clear
        </button>
      </div>
      <table>
        <thead>
          <tr>
            <th>Layers</th>
            <th>Width</th>
            <th>Params</th>
            <th>Regions</th>
            <th>Theory</th>
            <th>Accuracy</th>
            <th>Steps</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry, idx) => (
            <tr key={idx}>
              <td>{entry.depth}</td>
              <td>{entry.width}</td>
              <td>{entry.param_count}</td>
              <td className="highlight">{entry.actual_regions}</td>
              <td>{entry.theoretical_max}</td>
              <td>{(entry.accuracy * 100).toFixed(1)}%</td>
              <td>{entry.total_steps}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

