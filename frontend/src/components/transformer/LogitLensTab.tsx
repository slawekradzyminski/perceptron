import { useCallback } from "react";
import type { LogitLensResult } from "../../hooks/transformer/useGlassBoxApi";
import { TransformerInputSection } from "./TransformerInputSection";

type LogitLensTabProps = {
  inputText: string;
  onInputChange: (value: string) => void;
  loading: boolean;
  logitLens: LogitLensResult | null;
  onFetchLogitLens: (text: string, topK?: number) => void;
};

export function LogitLensTab({
  inputText,
  onInputChange,
  loading,
  logitLens,
  onFetchLogitLens,
}: LogitLensTabProps) {
  const handleAnalyze = useCallback(() => {
    onFetchLogitLens(inputText, 5);
  }, [inputText, onFetchLogitLens]);

  // Get color intensity for probability
  const getProbColor = (prob: number): string => {
    const intensity = Math.min(prob * 2, 1); // Scale up for visibility
    const green = Math.round(intensity * 200);
    return `rgba(34, ${100 + green}, 80, ${0.3 + intensity * 0.5})`;
  };

  return (
    <div className="logit-lens-tab">
      <div className="logit-lens-controls">
        <TransformerInputSection
          inputText={inputText}
          onInputChange={onInputChange}
        />
        <button
          className="analyze-button"
          onClick={handleAnalyze}
          disabled={loading || !inputText}
        >
          {loading ? "Analyzing..." : "Analyze Logit Lens"}
        </button>
      </div>

      {logitLens && (
        <div className="logit-lens-results">
          <div className="logit-lens-info">
            <span className="model-badge">{logitLens.model_name}</span>
            <span className="info-item">{logitLens.n_layers} layers</span>
            <span className="info-item">
              Predicting next token after: "{logitLens.tokens.join("")}"
            </span>
          </div>

          <div className="logit-lens-table-container">
            <table className="logit-lens-table">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Top Prediction</th>
                  <th>Confidence</th>
                  <th>Alternatives</th>
                </tr>
              </thead>
              <tbody>
                {logitLens.layers.map((layer) => {
                  const top = layer.predictions[0];
                  const rest = layer.predictions.slice(1);
                  return (
                    <tr key={layer.layer}>
                      <td className="layer-cell">
                        <span className="layer-name">{layer.layer_name}</span>
                      </td>
                      <td
                        className="prediction-cell"
                        style={{ backgroundColor: getProbColor(top.probability) }}
                      >
                        <span className="token-text">
                          {top.token.replace(/^\s/, "·")}
                        </span>
                      </td>
                      <td className="prob-cell">
                        <div className="prob-bar-container">
                          <div
                            className="prob-bar"
                            style={{ width: `${top.probability * 100}%` }}
                          />
                          <span className="prob-value">
                            {(top.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="alternatives-cell">
                        {rest.slice(0, 3).map((pred, i) => (
                          <span key={i} className="alt-token" title={`${(pred.probability * 100).toFixed(1)}%`}>
                            {pred.token.replace(/^\s/, "·")}
                          </span>
                        ))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="logit-lens-education">
            <h4>What is the Logit Lens?</h4>
            <p>
              The Logit Lens reveals what the model "believes" at each layer by
              applying the final output projection to intermediate hidden
              states.
            </p>
            <ul>
              <li>
                <strong>Early layers</strong> often show generic or uncertain
                predictions
              </li>
              <li>
                <strong>Middle layers</strong> start forming semantic
                understanding
              </li>
              <li>
                <strong>Later layers</strong> converge toward the final
                prediction
              </li>
            </ul>
            <p className="hint">
              This technique helps understand how transformers process
              information through depth (Chapter 7).
            </p>
          </div>
        </div>
      )}

      {!logitLens && !loading && (
        <div className="logit-lens-placeholder">
          <p>
            Enter text and click "Analyze Logit Lens" to see how predictions
            evolve through layers.
          </p>
          <p className="hint">
            The Logit Lens shows the model's "internal monologue" — what it
            thinks at each depth level (Chapter 7).
          </p>
        </div>
      )}
    </div>
  );
}
