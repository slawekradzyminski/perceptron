import { PromptLogitsCard } from "../components/gd/PromptLogitsCard";
import { TokenLossTableCard } from "../components/gd/TokenLossTableCard";
import { useGdApi } from "../hooks/gd/useGdApi";

export type GdPageProps = {
  apiBase: string;
};

export function GdPage({ apiBase }: GdPageProps) {
  const { tokenLosses, status, error, loading } = useGdApi(apiBase);

  return (
    <section className="panel lms-panel">
      <div className="lms-head">
        <div>
          <h2>Gradient Descent Exercises</h2>
        </div>
        <div className="gd-status-indicator">
          <span className={`gd-dot ${status?.ok ? "ok" : "bad"}`} />
          {status?.ok ? "Ollama connected" : "Ollama offline"}
        </div>
      </div>

      {loading && <p className="lms-empty">Loading Chapter 2 data...</p>}
      {error && <p className="diag-error">{error}</p>}

      {tokenLosses && (
        <div className="lms-exercises gd-layout">
          <PromptLogitsCard apiBase={apiBase} />
          <TokenLossTableCard apiBase={apiBase} examples={tokenLosses.examples} />
        </div>
      )}
    </section>
  );
}
