import { useState } from "react";
import type {
  BlockOperation,
  TraceResult,
} from "../../hooks/transformer/useTransformerApi";

type TransformerTraceTabProps = {
  inputText: string;
  loading: boolean;
  trace: TraceResult | null;
  onTrace: (nBlocks: number) => void;
};

export function TransformerTraceTab({
  inputText,
  loading,
  trace,
  onTrace,
}: TransformerTraceTabProps) {
  const [nBlocks, setNBlocks] = useState(12);
  const [hoveredOperation, setHoveredOperation] = useState<BlockOperation | null>(
    null,
  );

  return (
    <div className="trace-section">
      <div className="trace-controls">
        <label>
          Blocks to show:
          <input
            type="number"
            value={nBlocks}
            onChange={(e) =>
              setNBlocks(Math.min(96, Math.max(1, Number(e.target.value))))
            }
            min={1}
            max={96}
          />
        </label>
        <button onClick={() => onTrace(nBlocks)} disabled={loading || !inputText}>
          Show Trace
        </button>
      </div>

      {trace && (
        <div className="trace-result">
          <div className="trace-header">
            <div className="arch-info">
              <span className="info-item">
                <strong>Heads:</strong> {trace.n_heads}
              </span>
              <span className="info-item">
                <strong>d_head:</strong> {trace.d_head}
              </span>
              <span className="info-item">
                <strong>d_ff:</strong> {trace.d_ff}
              </span>
            </div>
          </div>

          <div className="block-stack">
            <div className="input-matrix">
              <strong>Input Embeddings</strong>
              <span className="shape">
                ({trace.token_count} × {trace.d_model})
              </span>
            </div>

            {trace.blocks.map((block) => (
              <div key={block.block} className="block-item">
                <div className="block-header">
                  <span className="block-num">Block {block.block}</span>
                  <span className="block-shape">
                    [{block.input_shape[0]} × {block.input_shape[1]}] → [{
                      block.output_shape[0]
                    } × {block.output_shape[1]}]
                  </span>
                </div>
                <div className="operations-list">
                  {block.operations.map((op, idx) => (
                    <div
                      key={idx}
                      className={`operation ${
                        hoveredOperation === op ? "hovered" : ""
                      }`}
                      onMouseEnter={() => setHoveredOperation(op)}
                      onMouseLeave={() => setHoveredOperation(null)}
                    >
                      <span className="op-name">{op.name}</span>
                      <span className="op-formula">{op.formula}</span>
                      {hoveredOperation === op && (
                        <div className="operation-tooltip">
                          <h5>{op.name}</h5>
                          <code className="formula">{op.formula}</code>
                          {op.details && (
                            <div className="op-details">
                              {op.details.n_heads && (
                                <p>
                                  <strong>Heads:</strong> {op.details.n_heads} ×{" "}
                                  {op.details.d_head} dims each
                                </p>
                              )}
                              {op.details.attention_scores_shape && (
                                <p>
                                  <strong>Attention scores:</strong> [
                                  {op.details.attention_scores_shape.join(" × ")}]
                                </p>
                              )}
                              {op.details.hidden_dim && (
                                <p>
                                  <strong>Hidden expansion:</strong> {trace.d_model} →{" "}
                                  {op.details.hidden_dim} → {trace.d_model}
                                </p>
                              )}
                              <p className="explanation">
                                {op.details.explanation}
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            <div className="output-matrix">
              <strong>Final Output</strong>
              <span className="shape">
                ({trace.final_shape[0]} × {trace.final_shape[1]})
              </span>
              <span className="hint">→ Last column predicts next token</span>
            </div>
          </div>

          <p className="explanation">{trace.explanation}</p>
        </div>
      )}
    </div>
  );
}
