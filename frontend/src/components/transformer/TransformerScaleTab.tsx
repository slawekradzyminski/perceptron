import { useState } from "react";
import type {
  GrowthData,
  ScaleComparison,
  ScaleModel,
} from "../../hooks/transformer/useTransformerApi";

function formatParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toString();
}

type TransformerScaleTabProps = {
  scaleModels: ScaleModel[];
  comparison: ScaleComparison | null;
  growth: GrowthData | null;
  loading: boolean;
  onCompare: (model1: string, model2: string) => void;
};

export function TransformerScaleTab({
  scaleModels,
  comparison,
  growth,
  loading,
  onCompare,
}: TransformerScaleTabProps) {
  const [model1, setModel1] = useState("AlexNet");
  const [model2, setModel2] = useState("GPT-4");

  const resolvedModel1 = scaleModels.some((model) => model.name === model1)
    ? model1
    : scaleModels[0]?.name ?? model1;
  const resolvedModel2 = scaleModels.some((model) => model.name === model2)
    ? model2
    : scaleModels[1]?.name ?? scaleModels[0]?.name ?? model2;

  const maxLog = growth
    ? Math.max(...growth.data.map((model) => model.log_params))
    : 1;

  return (
    <div className="scale-section">
      <div className="scale-controls">
        <select value={resolvedModel1} onChange={(e) => setModel1(e.target.value)}>
          {scaleModels.map((model) => (
            <option key={model.name} value={model.name}>
              {model.name}
            </option>
          ))}
        </select>
        <span>vs</span>
        <select value={resolvedModel2} onChange={(e) => setModel2(e.target.value)}>
          {scaleModels.map((model) => (
            <option key={model.name} value={model.name}>
              {model.name}
            </option>
          ))}
        </select>
        <button
          onClick={() => onCompare(resolvedModel1, resolvedModel2)}
          disabled={loading}
        >
          Compare
        </button>
      </div>

      {comparison && (
        <div className="comparison-result">
          <div className="model-cards">
            <div className="model-card">
              <h4>{comparison.model1.name}</h4>
              <p className="params">{comparison.model1.params_formatted}</p>
              <p>{comparison.model1.year}</p>
            </div>
            <div className="ratio-display">
              <span>{comparison.ratio_formatted}</span>
            </div>
            <div className="model-card">
              <h4>{comparison.model2.name}</h4>
              <p className="params">{comparison.model2.params_formatted}</p>
              <p>{comparison.model2.year}</p>
            </div>
          </div>
          <p className="explanation">{comparison.explanation}</p>
        </div>
      )}

      {growth && (
        <div className="growth-section">
          <h4>Model Growth Over Time</h4>
          <div className="growth-chart-container">
            <div className="y-axis">
              <span className="y-label">1T</span>
              <span className="y-label">1B</span>
              <span className="y-label">1M</span>
              <span className="y-label">1K</span>
            </div>
            <div className="growth-chart">
              {growth.data.map((model) => {
                const height = (model.log_params / maxLog) * 180;
                return (
                  <div
                    key={model.name}
                    className={`growth-bar ${model.type.toLowerCase()}`}
                    style={{ height: `${height}px` }}
                  >
                    <span className="bar-value">{formatParams(model.params)}</span>
                    <span className="bar-label">{model.name}</span>
                    <span className="bar-year">{model.year}</span>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="chart-legend">
            <span className="legend-item cnn">■ CNN</span>
            <span className="legend-item transformer">■ Transformer</span>
          </div>
          <p className="insight">{growth.insight}</p>
        </div>
      )}

      <div className="scale-table">
        <h4>All Models</h4>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Year</th>
              <th>Parameters</th>
              <th>Type</th>
            </tr>
          </thead>
          <tbody>
            {scaleModels.map((model) => (
              <tr key={model.name}>
                <td>{model.name}</td>
                <td>{model.year}</td>
                <td>{model.params_formatted}</td>
                <td className={model.type.toLowerCase()}>{model.type}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
