import { PromptLogitsCard } from "../components/gd/PromptLogitsCard";
import { TokenLossTableCard } from "../components/gd/TokenLossTableCard";
import { useGdApi } from "../hooks/gd/useGdApi";

export type GdPageProps = {
  apiBase: string;
};

export function GdPage({ apiBase }: GdPageProps) {
  const { tokenLosses, status, error, loading } = useGdApi(apiBase);

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
          <span className={`gd-dot ${status?.ok ? "ok" : "bad"}`} />
          {status?.ok ? "Ollama connected" : "Ollama offline"}
        </div>
      </div>

      {loading && <p className="gd-loading">Loading Chapter 2 data...</p>}
      {error && <p className="diag-error">{error}</p>}

      {tokenLosses && (
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
