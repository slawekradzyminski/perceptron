import { useState, useCallback, useMemo } from "react";
import type { AttentionResult } from "../../hooks/transformer/useGlassBoxApi";
import { AttentionHeatmap } from "./AttentionHeatmap";
import { TransformerInputSection } from "./TransformerInputSection";

type AttentionTabProps = {
  inputText: string;
  onInputChange: (value: string) => void;
  loading: boolean;
  attention: AttentionResult | null;
  onFetchAttention: (text: string) => void;
};

export function AttentionTab({
  inputText,
  onInputChange,
  loading,
  attention,
  onFetchAttention,
}: AttentionTabProps) {
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);

  const handleAnalyze = useCallback(() => {
    onFetchAttention(inputText);
    setSelectedLayer(0);
    setSelectedHead(0);
  }, [inputText, onFetchAttention]);

  // Get the attention matrix for the selected layer/head
  const selectedAttention = useMemo(() => {
    if (!attention) return null;
    return attention.attentions[selectedLayer]?.[selectedHead] ?? null;
  }, [attention, selectedLayer, selectedHead]);

  return (
    <div className="attention-tab">
      <div className="attention-controls">
        <TransformerInputSection
          inputText={inputText}
          onInputChange={onInputChange}
        />
        <button
          className="analyze-button"
          onClick={handleAnalyze}
          disabled={loading || !inputText}
        >
          {loading ? "Analyzing..." : "Analyze Attention"}
        </button>
      </div>

      {attention && (
        <div className="attention-results">
          <div className="attention-info">
            <span className="model-badge">{attention.model_name}</span>
            <span className="info-item">
              {attention.n_layers} layers × {attention.n_heads} heads
            </span>
            <span className="info-item">{attention.seq_len} tokens</span>
          </div>

          <div className="head-selector">
            <div className="selector-group">
              <label>Layer</label>
              <select
                value={selectedLayer}
                onChange={(e) => setSelectedLayer(Number(e.target.value))}
              >
                {Array.from({ length: attention.n_layers }, (_, i) => (
                  <option key={i} value={i}>
                    Layer {i + 1}
                  </option>
                ))}
              </select>
            </div>

            <div className="selector-group">
              <label>Head</label>
              <select
                value={selectedHead}
                onChange={(e) => setSelectedHead(Number(e.target.value))}
              >
                {Array.from({ length: attention.n_heads }, (_, i) => (
                  <option key={i} value={i}>
                    Head {i + 1}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="attention-display">
            <h4>
              Attention Pattern — Layer {selectedLayer + 1}, Head{" "}
              {selectedHead + 1}
            </h4>
            {selectedAttention && (
              <AttentionHeatmap
                matrix={selectedAttention}
                tokens={attention.tokens}
              />
            )}
          </div>

          <div className="attention-education">
            <h4>How to Read This</h4>
            <ul>
              <li>
                <strong>Rows</strong> = query positions (which token is
                "asking")
              </li>
              <li>
                <strong>Columns</strong> = key positions (which token is being
                "attended to")
              </li>
              <li>
                <strong>Brighter</strong> = stronger attention weight
              </li>
              <li>
                <strong>Diagonal pattern</strong> = "previous token" head
              </li>
              <li>
                <strong>First column</strong> = attention to BOS/start token
              </li>
            </ul>
          </div>
        </div>
      )}

      {!attention && !loading && (
        <div className="attention-placeholder">
          <p>
            Enter text above and click "Analyze Attention" to visualize how
            tokens attend to each other.
          </p>
          <p className="hint">
            This reveals the internal workings of the transformer — the "Glass
            Box" view (Chapter 8).
          </p>
        </div>
      )}
    </div>
  );
}
