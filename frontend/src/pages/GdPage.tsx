import { LossComparisonCard } from "../components/gd/LossComparisonCard";
import { TokenLossTableCard } from "../components/gd/TokenLossTableCard";
import { useGdApi } from "../hooks/gd/useGdApi";

export type GdPageProps = {
  apiBase: string;
};

export function GdPage({ apiBase }: GdPageProps) {
  const { lossCurves, tokenLosses, error, loading } = useGdApi(apiBase);

  return (
    <section className="panel lms-panel">
      <div className="lms-head">
        <div>
          <h2>Gradient Descent Exercises</h2>
          <p className="panel-subtle">
            Cross-entropy vs L1 and token-level loss tables from Chapter 2.
          </p>
        </div>
      </div>

      {loading && <p className="lms-empty">Loading Chapter 2 data...</p>}
      {error && <p className="diag-error">{error}</p>}

      {lossCurves && tokenLosses && (
        <div className="lms-exercises">
          <LossComparisonCard points={lossCurves.points} />
          <TokenLossTableCard examples={tokenLosses.examples} />
        </div>
      )}
    </section>
  );
}
