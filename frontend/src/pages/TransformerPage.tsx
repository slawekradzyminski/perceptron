import { useEffect, useState, useCallback } from "react";
import { useTransformerApi, TokenEmbedding, BlockOperation } from "../hooks/transformer/useTransformerApi";
import { TransformerEducation } from "../components/education/TransformerEducation";
import "../styles/transformer.css";
import "../styles/education.css";

// Token colors for visualization
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

// Format embedding values for display
function formatEmbedding(values: number[], maxDisplay: number = 10): string {
  const displayed = values.slice(0, maxDisplay);
  const formatted = displayed.map(v => v.toFixed(4)).join(", ");
  if (values.length > maxDisplay) {
    return `[${formatted}, ... (${values.length - maxDisplay} more)]`;
  }
  return `[${formatted}]`;
}

// Format number with SI suffix
function formatParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toString();
}

export function TransformerPage({ apiBase }: { apiBase: string }) {
  const {
    tokens,
    embedInfo,
    trace,
    chatMessages,
    streamingContent,
    ollamaStatus,
    scaleModels,
    comparison,
    growth,
    error,
    loading,
    generating,
    tokenize,
    getEmbedInfo,
    getTrace,
    sendChat,
    resetChat,
    fetchOllamaStatus,
    fetchScaleModels,
    compareModels,
    fetchGrowth,
  } = useTransformerApi(apiBase);

  const [inputText, setInputText] = useState("The quick brown fox jumps");
  const [chatInput, setChatInput] = useState("");
  const [activeTab, setActiveTab] = useState<
    "tokenize" | "trace" | "generate" | "scale"
  >("tokenize");
  const [nBlocks, setNBlocks] = useState(12);
  const [model1, setModel1] = useState("AlexNet");
  const [model2, setModel2] = useState("GPT-4");
  const [hoveredToken, setHoveredToken] = useState<TokenEmbedding | null>(null);
  const [hoveredMatrixCell, setHoveredMatrixCell] = useState<{row: number, col: number, value: number} | null>(null);
  const [hoveredOperation, setHoveredOperation] = useState<BlockOperation | null>(null);

  const ollamaOnline = ollamaStatus?.ok === true && ollamaStatus?.model_available === true;

  // Initialize
  useEffect(() => {
    void fetchScaleModels();
    void fetchGrowth();
    void fetchOllamaStatus();
  }, [fetchScaleModels, fetchGrowth, fetchOllamaStatus]);

  const handleTokenize = useCallback(() => {
    void tokenize(inputText);
    void getEmbedInfo(inputText);
  }, [tokenize, getEmbedInfo, inputText]);

  const handleTrace = useCallback(() => {
    void getTrace(inputText, nBlocks);
  }, [getTrace, inputText, nBlocks]);

  const handleSendChat = useCallback(() => {
    if (chatInput.trim()) {
      void sendChat(chatInput);
      setChatInput("");
    }
  }, [sendChat, chatInput]);

  const handleCompare = useCallback(() => {
    void compareModels(model1, model2);
  }, [compareModels, model1, model2]);

  return (
    <section className="panel transformer-panel">
      <header className="transformer-header">
        <h2>Transformer Flow</h2>
        <p className="subtitle">
          Understanding GPT — From text to tokens to predictions
        </p>
      </header>

      <div className="transformer-grid">
        {/* Input section */}
        <div className="transformer-input">
          <label>Enter text:</label>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to tokenize..."
            rows={3}
          />
        </div>

        {/* Tab navigation */}
        <div className="tab-nav">
          <button
            className={activeTab === "tokenize" ? "active" : ""}
            onClick={() => setActiveTab("tokenize")}
          >
            Tokenization
          </button>
          <button
            className={activeTab === "trace" ? "active" : ""}
            onClick={() => setActiveTab("trace")}
          >
            Block Trace
          </button>
          <button
            className={activeTab === "generate" ? "active" : ""}
            onClick={() => setActiveTab("generate")}
          >
            Generation
          </button>
          <button
            className={activeTab === "scale" ? "active" : ""}
            onClick={() => setActiveTab("scale")}
          >
            Model Scale
          </button>
        </div>

        {/* Tab content */}
        <div className="tab-content">
          {/* Tokenization tab */}
          {activeTab === "tokenize" && (
            <div className="tokenize-section">
              <div className="section-header">
                <button onClick={handleTokenize} disabled={loading || !inputText}>
                  {loading ? "Loading..." : "Tokenize & Embed"}
                </button>
                <div className="model-info-badge">
                  <span className="model-label">Embeddings from:</span>
                  <span className="model-name">GPT-2</span>
                  <span className="model-note">(768-dim learned representations)</span>
                </div>
              </div>

              {/* Loading indicator */}
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
                  <h4>
                    Tokens ({tokens.token_count})
                  </h4>
                  <div className="token-display">
                    {embedInfo ? embedInfo.tokens.map((token, i) => (
                      <span
                        key={i}
                        className={`token ${hoveredToken?.id === token.id ? "hovered" : ""}`}
                        style={{
                          backgroundColor:
                            TOKEN_COLORS[i % TOKEN_COLORS.length],
                        }}
                        onMouseEnter={() => setHoveredToken(token)}
                        onMouseLeave={() => setHoveredToken(null)}
                      >
                        {token.text || "⎵"}
                      </span>
                    )) : tokens.tokens.map((token, i) => (
                      <span
                        key={i}
                        className="token"
                        style={{
                          backgroundColor:
                            TOKEN_COLORS[i % TOKEN_COLORS.length],
                        }}
                        title={`ID: ${token.id}`}
                      >
                        {token.text || "⎵"}
                      </span>
                    ))}
                  </div>
                  <div className="token-ids">
                    <strong>Token IDs:</strong>{" "}
                    [{tokens.tokens.map((t) => t.id).join(", ")}]
                  </div>

                  {/* Embedding tooltip */}
                  {hoveredToken && (
                    <div className="embedding-tooltip">
                      <div className="tooltip-header">
                        <span className="token-text">"{hoveredToken.text || "⎵"}"</span>
                        <span className="token-id">ID: {hoveredToken.id}</span>
                      </div>
                      <div className="embedding-stats">
                        <span>min: {hoveredToken.min}</span>
                        <span>max: {hoveredToken.max}</span>
                        <span>μ: {hoveredToken.mean}</span>
                        <span>σ: {hoveredToken.std}</span>
                      </div>
                      <div className="embedding-preview">
                        <strong>First 10 dims:</strong>
                        <code>{formatEmbedding(hoveredToken.embedding_preview, 10)}</code>
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

                  {/* Interactive matrix preview - first 8 dimensions */}
                  <div className="matrix-preview">
                    <div className="matrix-header">
                      <span className="row-label"></span>
                      {[0,1,2,3,4,5,6,7].map(d => (
                        <span key={d} className="col-label">d{d}</span>
                      ))}
                      <span className="col-label">...</span>
                    </div>
                    {embedInfo.tokens.map((token, row) => (
                      <div key={row} className="matrix-row">
                        <span
                          className="row-label"
                          style={{ backgroundColor: TOKEN_COLORS[row % TOKEN_COLORS.length] }}
                        >
                          {token.text || "⎵"}
                        </span>
                        {token.embedding_preview.slice(0, 8).map((val, col) => (
                          <span
                            key={col}
                            className={`matrix-cell ${hoveredMatrixCell?.row === row && hoveredMatrixCell?.col === col ? "hovered" : ""}`}
                            style={{
                              backgroundColor: `rgba(138, 43, 226, ${Math.abs(val) * 0.8})`,
                              color: Math.abs(val) > 0.5 ? "white" : "inherit"
                            }}
                            onMouseEnter={() => setHoveredMatrixCell({row, col, value: val})}
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
                        Dim {hoveredMatrixCell.col}: <strong>{hoveredMatrixCell.value.toFixed(6)}</strong>
                      </div>
                    )}
                  </div>

                  <p className="explanation">{embedInfo.explanation}</p>
                </div>
              )}
            </div>
          )}

          {/* Block Trace tab */}
          {activeTab === "trace" && (
            <div className="trace-section">
              <div className="trace-controls">
                <label>
                  Blocks to show:
                  <input
                    type="number"
                    value={nBlocks}
                    onChange={(e) =>
                      setNBlocks(Math.min(96, Math.max(1, +e.target.value)))
                    }
                    min={1}
                    max={96}
                  />
                </label>
                <button onClick={handleTrace} disabled={loading || !inputText}>
                  Show Trace
                </button>
              </div>

              {trace && (
                <div className="trace-result">
                  <div className="trace-header">
                    <div className="arch-info">
                      <span className="info-item"><strong>Heads:</strong> {trace.n_heads}</span>
                      <span className="info-item"><strong>d_head:</strong> {trace.d_head}</span>
                      <span className="info-item"><strong>d_ff:</strong> {trace.d_ff}</span>
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
                            [{block.input_shape[0]} × {block.input_shape[1]}] → [{block.output_shape[0]} × {block.output_shape[1]}]
                          </span>
                        </div>
                        <div className="operations-list">
                          {block.operations.map((op, idx) => (
                            <div
                              key={idx}
                              className={`operation ${hoveredOperation === op ? "hovered" : ""}`}
                              onMouseEnter={() => setHoveredOperation(op)}
                              onMouseLeave={() => setHoveredOperation(null)}
                            >
                              <span className="op-name">{op.name}</span>
                              <span className="op-formula">{op.formula}</span>
                              {/* Inline tooltip */}
                              {hoveredOperation === op && (
                                <div className="operation-tooltip">
                                  <h5>{op.name}</h5>
                                  <code className="formula">{op.formula}</code>
                                  {op.details && (
                                    <div className="op-details">
                                      {op.details.n_heads && (
                                        <p><strong>Heads:</strong> {op.details.n_heads} × {op.details.d_head} dims each</p>
                                      )}
                                      {op.details.attention_scores_shape && (
                                        <p><strong>Attention scores:</strong> [{op.details.attention_scores_shape.join(" × ")}]</p>
                                      )}
                                      {op.details.hidden_dim && (
                                        <p><strong>Hidden expansion:</strong> {trace.d_model} → {op.details.hidden_dim} → {trace.d_model}</p>
                                      )}
                                      <p className="explanation">{op.details.explanation}</p>
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
          )}

          {/* Chat tab */}
          {activeTab === "generate" && (
            <div className="chat-section">
              {/* Ollama status indicator */}
              <div className="ollama-status-row">
                <div className={`ollama-status ${ollamaOnline ? "online" : "offline"}`}>
                  <span className={`status-dot ${ollamaOnline ? "ok" : "bad"}`} />
                  {ollamaOnline ? "Ollama connected" : "Ollama offline"}
                </div>
                {ollamaOnline && (
                  <>
                    <div className="model-info-badge llama">
                      <span className="model-label">Model:</span>
                      <span className="model-name">{ollamaStatus?.model}</span>
                    </div>
                    {chatMessages.length > 0 && (
                      <button onClick={resetChat} className="reset-btn">
                        Reset Chat
                      </button>
                    )}
                  </>
                )}
              </div>

              {/* Offline notice */}
              {!ollamaOnline && (
                <div className="ollama-offline-notice">
                  <div className="offline-icon">⚡</div>
                  <h3>Ollama Required</h3>
                  <p>
                    This tab requires Ollama to be running locally for live LLM chat.
                  </p>
                  <div className="offline-instructions">
                    <p><strong>To get started:</strong></p>
                    <ol>
                      <li>Install Ollama from <code>ollama.ai</code></li>
                      <li>Run <code>ollama serve</code> in a terminal</li>
                      <li>Pull the model: <code>ollama pull llama3.2:1b</code></li>
                      <li>Refresh this page</li>
                    </ol>
                    <p><strong>Or with Docker:</strong></p>
                    <code>docker run -d -p 11434:11434 ollama/ollama</code>
                  </div>
                </div>
              )}

              {/* Chat interface - only show when Ollama is online */}
              {ollamaOnline && (
                <div className="chat-container">
                  {/* Chat messages */}
                  <div className="chat-messages">
                    {chatMessages.length === 0 && !generating && (
                      <div className="chat-empty">
                        <div className="empty-icon">💬</div>
                        <p>Start a conversation with {ollamaStatus?.model}</p>
                        <p className="empty-hint">Try asking a question or giving a prompt</p>
                      </div>
                    )}

                    {chatMessages.map((msg, idx) => (
                      <div key={idx} className={`chat-message ${msg.role}`}>
                        <div className="message-label">
                          {msg.role === "user" ? "You" : ollamaStatus?.model}
                        </div>
                        <div className="message-content">{msg.content}</div>
                      </div>
                    ))}

                    {/* Streaming response */}
                    {generating && streamingContent && (
                      <div className="chat-message assistant streaming">
                        <div className="message-label">{ollamaStatus?.model}</div>
                        <div className="message-content">
                          {streamingContent}
                          <span className="cursor">▊</span>
                        </div>
                      </div>
                    )}

                    {/* Loading indicator when no content yet */}
                    {generating && !streamingContent && (
                      <div className="chat-message assistant streaming">
                        <div className="message-label">{ollamaStatus?.model}</div>
                        <div className="message-content typing">
                          <span className="dot"></span>
                          <span className="dot"></span>
                          <span className="dot"></span>
                        </div>
                      </div>
                    )}

                    {error && (
                      <div className="chat-error">
                        <strong>⚠️ Error:</strong> {error}
                      </div>
                    )}
                  </div>

                  {/* Chat input */}
                  <div className="chat-input-area">
                    <textarea
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
                      rows={2}
                      disabled={generating}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSendChat();
                        }
                      }}
                    />
                    <button
                      onClick={handleSendChat}
                      disabled={generating || !chatInput.trim()}
                      className="send-btn"
                    >
                      {generating ? "..." : "Send"}
                    </button>
                  </div>

                  <p className="chat-hint">
                    Responses stream token-by-token. The model maintains conversation history.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Scale tab */}
          {activeTab === "scale" && (
            <div className="scale-section">
              <div className="scale-controls">
                <select value={model1} onChange={(e) => setModel1(e.target.value)}>
                  {scaleModels.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                    </option>
                  ))}
                </select>
                <span>vs</span>
                <select value={model2} onChange={(e) => setModel2(e.target.value)}>
                  {scaleModels.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}
                    </option>
                  ))}
                </select>
                <button onClick={handleCompare} disabled={loading}>
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
                        const maxLog = Math.max(...growth.data.map(d => d.log_params));
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
          )}
        </div>
      </div>

      <TransformerEducation />

      {error && <p className="transformer-error">{error}</p>}
    </section>
  );
}
