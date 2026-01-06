import { PromptLogitsCard } from "../components/gd/PromptLogitsCard";
import { TokenLossTableCard } from "../components/gd/TokenLossTableCard";
import { useGdApi } from "../hooks/gd/useGdApi";

export type GdPageProps = {
  apiBase: string;
};

export function GdPage({ apiBase }: GdPageProps) {
  const { tokenLosses, status, error, loading } = useGdApi(apiBase);

  const ollamaOnline = status?.ok === true;

  return (
    <section className="panel gd-panel">
      <div className="gd-head">
        <div>
          <h2>Gradient Descent Exercises</h2>
          <p className="panel-subtle">
            Explore token probabilities and loss functions for language model predictions.
          </p>
        </div>
        <div className="gd-status-indicator">
          <span className={`gd-dot ${ollamaOnline ? "ok" : "bad"}`} />
          {ollamaOnline ? "Ollama connected" : "Ollama offline"}
        </div>
      </div>

      {!ollamaOnline && !loading && (
        <div className="gd-offline-notice">
          <div className="gd-offline-icon">⚡</div>
          <h3>Ollama Required</h3>
          <p>
            This page requires Ollama to be running locally for live LLM inference.
          </p>
          <div className="gd-offline-instructions">
            <p><strong>To get started:</strong></p>
            <ol>
              <li>Install Ollama from <code>ollama.ai</code></li>
              <li>Run <code>ollama serve</code> in a terminal</li>
              <li>Pull a model: <code>ollama pull llama3.2</code></li>
              <li>Refresh this page</li>
            </ol>
          </div>
        </div>
      )}

      {loading && <p className="gd-loading">Loading Chapter 2 data...</p>}
      {error && ollamaOnline && <p className="diag-error">{error}</p>}

      {ollamaOnline && tokenLosses && (
        <div className="gd-exercises">
          {/* Row 1: Next-token logits */}
          <div className="gd-row">
            <PromptLogitsCard apiBase={apiBase} />
          </div>

          {/* Row 2: Token loss table */}
          <div className="gd-row">
            <TokenLossTableCard apiBase={apiBase} examples={tokenLosses.examples} />
          </div>
        </div>
      )}
    </section>
  );
}
