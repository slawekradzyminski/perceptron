import { useState, useMemo, useCallback } from "react";
import type {
  EmbedResult,
  TokenEmbedding,
  TokenizeResult,
} from "../../hooks/transformer/useTransformerApi";

const TOKEN_COLORS = [
  "#FF6B6B",
  "#4ECDC4",
  "#45B7D1",
  "#96CEB4",
  "#FFEAA7",
  "#DDA0DD",
  "#98D8C8",
  "#F7DC6F",
  "#BB8FCE",
  "#85C1E9",
];

function formatEmbedding(values: number[], maxDisplay: number = 10): string {
  const displayed = values.slice(0, maxDisplay);
  const formatted = displayed.map((v) => v.toFixed(4)).join(", ");
  if (values.length > maxDisplay) {
    return `[${formatted}, ... (${values.length - maxDisplay} more)]`;
  }
  return `[${formatted}]`;
}

type TransformerTokenizeTabProps = {
  inputText: string;
  loading: boolean;
  tokens: TokenizeResult | null;
  embedInfo: EmbedResult | null;
  onTokenize: () => void;
};

export function TransformerTokenizeTab({
  inputText,
  loading,
  tokens,
  embedInfo,
  onTokenize,
}: TransformerTokenizeTabProps) {
  const [selectedToken, setSelectedToken] = useState<TokenEmbedding | null>(null);
  const [showFullEmbedding, setShowFullEmbedding] = useState(false);
  const [hoveredMatrixCell, setHoveredMatrixCell] = useState<{
    row: number;
    col: number;
    value: number;
  } | null>(null);

  const toggleFullEmbedding = useCallback(() => {
    setShowFullEmbedding((prev) => !prev);
  }, []);

  const handleTokenClick = useCallback((token: TokenEmbedding) => {
    if (selectedToken?.id === token.id) {
      setSelectedToken(null);
      setShowFullEmbedding(false);
    } else {
      setSelectedToken(token);
      setShowFullEmbedding(false);
    }
  }, [selectedToken]);

  // Calculate width for first column based on longest token
  const rowLabelWidth = useMemo(() => {
    if (!embedInfo) return 60;
    const longestToken = Math.max(
      ...embedInfo.tokens.map((t) => (t.text || "⎵").length)
    );
    // Approximate: 8px per character + 16px padding
    return Math.min(Math.max(longestToken * 8 + 16, 60), 120);
  }, [embedInfo]);

  return (
    <div className="tokenize-section">
      <div className="section-header">
        <button onClick={onTokenize} disabled={loading || !inputText}>
          {loading ? "Loading..." : "Tokenize & Embed"}
        </button>
        <div className="model-info-badge">
          <span className="model-label">Embeddings from:</span>
          <span className="model-name">GPT-2</span>
          <span className="model-note">(768-dim learned representations)</span>
        </div>
      </div>

      {loading && !tokens && (
        <div className="loading-indicator">
          <div className="loading-spinner"></div>
          <div className="loading-text">
            <strong>Loading GPT-2 model...</strong>
            <p>First load downloads ~500MB. This only happens once.</p>
          </div>
        </div>
      )}

      {tokens && (
        <div className="token-result">
          <h4>Tokens ({tokens.token_count})</h4>
          <div className="token-display">
            {embedInfo
              ? embedInfo.tokens.map((token, i) => (
                  <span
                    key={i}
                    className={`token clickable ${
                      selectedToken?.id === token.id ? "selected" : ""
                    }`}
                    style={{
                      backgroundColor: TOKEN_COLORS[i % TOKEN_COLORS.length],
                    }}
                    onClick={() => handleTokenClick(token)}
                    title="Click to view embedding"
                  >
                    {token.text || "⎵"}
                  </span>
                ))
              : tokens.tokens.map((token, i) => (
                  <span
                    key={i}
                    className="token"
                    style={{
                      backgroundColor: TOKEN_COLORS[i % TOKEN_COLORS.length],
                    }}
                    title={`ID: ${token.id}`}
                  >
                    {token.text || "⎵"}
                  </span>
                ))}
          </div>
          <div className="token-ids">
            <strong>Token IDs:</strong> [{tokens.tokens.map((t) => t.id).join(", ")}] 
          </div>

          {selectedToken && (
            <div className={`embedding-tooltip ${showFullEmbedding ? "expanded" : ""}`}>
              <div className="tooltip-header">
                <span className="token-text">
                  "{selectedToken.text || "⎵"}"
                </span>
                <span className="token-id">ID: {selectedToken.id}</span>
                <button
                  className="close-tooltip-btn"
                  onClick={() => { setSelectedToken(null); setShowFullEmbedding(false); }}
                  title="Close"
                >
                  ✕
                </button>
              </div>
              <div className="embedding-stats">
                <span>min: {selectedToken.min}</span>
                <span>max: {selectedToken.max}</span>
                <span>μ: {selectedToken.mean}</span>
                <span>σ: {selectedToken.std}</span>
              </div>
              <div className="embedding-preview">
                <div className="preview-header">
                  <strong>{showFullEmbedding ? `All 768 dims:` : `First 10 dims:`}</strong>
                  <button
                    className="toggle-full-btn"
                    onClick={toggleFullEmbedding}
                    title={showFullEmbedding ? "Show less" : "Show all 768 dimensions"}
                  >
                    {showFullEmbedding ? "▲ Show less" : "▼ Show all 768"}
                  </button>
                </div>
                <code className={showFullEmbedding ? "full-embedding" : ""}>
                  {showFullEmbedding
                    ? `[${selectedToken.embedding.map((v) => v.toFixed(4)).join(", ")}]`
                    : formatEmbedding(selectedToken.embedding_preview, 10)}
                </code>
              </div>
            </div>
          )}
        </div>
      )}

      {embedInfo && (
        <div className="embed-info">
          <h4>Embedding Matrix</h4>
          <div className="matrix-viz">
            <div className="matrix-shape">
              <span className="dim">{embedInfo.token_count}</span>
              <span className="label">tokens</span>
              <span className="times">×</span>
              <span className="dim">{embedInfo.d_model}</span>
              <span className="label">dimensions</span>
            </div>
          </div>

          <div className="matrix-preview">
            <div className="matrix-header">
              <span className="row-label" style={{ width: rowLabelWidth }}></span>
              {[0, 1, 2, 3, 4, 5, 6, 7].map((d) => (
                <span key={d} className="col-label">
                  d{d}
                </span>
              ))}
              <span className="col-label">...</span>
            </div>
            {embedInfo.tokens.map((token, row) => (
              <div key={row} className="matrix-row">
                <span
                  className="row-label"
                  style={{
                    backgroundColor: TOKEN_COLORS[row % TOKEN_COLORS.length],
                    width: rowLabelWidth,
                  }}
                  title={token.text || "⎵"}
                >
                  {token.text || "⎵"}
                </span>
                {token.embedding_preview.slice(0, 8).map((val, col) => (
                  <span
                    key={col}
                    className={`matrix-cell ${
                      hoveredMatrixCell?.row === row &&
                      hoveredMatrixCell?.col === col
                        ? "hovered"
                        : ""
                    }`}
                    style={{
                      backgroundColor: `rgba(138, 43, 226, ${
                        Math.abs(val) * 0.8
                      })`,
                      color: Math.abs(val) > 0.5 ? "white" : "inherit",
                    }}
                    onMouseEnter={() =>
                      setHoveredMatrixCell({ row, col, value: val })
                    }
                    onMouseLeave={() => setHoveredMatrixCell(null)}
                  >
                    {val.toFixed(2)}
                  </span>
                ))}
                <span className="matrix-cell ellipsis">...</span>
              </div>
            ))}
            {hoveredMatrixCell && (
              <div className="matrix-cell-tooltip">
                Token "{embedInfo.tokens[hoveredMatrixCell.row].text || "⎵"}", 
                Dim {hoveredMatrixCell.col}: 
                <strong>{hoveredMatrixCell.value.toFixed(6)}</strong>
              </div>
            )}
          </div>

          <p className="explanation">{embedInfo.explanation}</p>
        </div>
      )}
    </div>
  );
}
