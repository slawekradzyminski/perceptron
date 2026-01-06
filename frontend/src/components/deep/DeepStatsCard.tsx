import type { DeepArchitecture, DeepMetrics, DeepRegions } from "../../hooks/deep/useDeepApi";

type DeepStatsCardProps = {
  architecture: DeepArchitecture | null;
  metrics: DeepMetrics | null;
  regions: DeepRegions | null;
  epoch: number;
  totalSteps: number;
};

export function DeepStatsCard({
  architecture,
  metrics,
  regions,
  epoch,
  totalSteps,
}: DeepStatsCardProps) {
  return (
    <div className="deep-stats">
      <h3>Training Stats</h3>

      <div className="stats-grid">
        <div className="stat-item">
          <span className="stat-label">Epoch</span>
          <span className="stat-value">{epoch}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Steps</span>
          <span className="stat-value">{totalSteps}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Loss</span>
          <span className="stat-value">{metrics?.loss.toFixed(4) ?? "—"}</span>
        </div>
        <div className="stat-item accuracy">
          <span className="stat-label">Accuracy</span>
          <span className="stat-value">
            {metrics ? `${(metrics.accuracy * 100).toFixed(1)}%` : "—"}
          </span>
        </div>
      </div>

      <h3>Region Analysis</h3>
      <div className="stats-grid">
        <div className="stat-item regions">
          <span className="stat-label">Actual Regions</span>
          <span className="stat-value highlight">{regions?.count ?? "—"}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Theory Max</span>
          <span className="stat-value">{regions?.theoretical_max ?? "—"}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Efficiency</span>
          <span className="stat-value">
            {regions ? `${(regions.efficiency * 100).toFixed(1)}%` : "—"}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Parameters</span>
          <span className="stat-value">{architecture?.param_count ?? "—"}</span>
        </div>
      </div>
    </div>
  );
}

