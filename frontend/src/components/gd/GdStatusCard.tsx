import type { OllamaStatusResponse } from "../../types";

export function GdStatusCard({ status }: { status: OllamaStatusResponse | null }) {
  if (!status) {
    return (
      <div className="lms-math gd-status">
        <h3>Ollama status</h3>
        <p>Loading status...</p>
      </div>
    );
  }

  if (!status.ok) {
    return (
      <div className="lms-math gd-status">
        <h3>Ollama status</h3>
        <p className="diag-error">Unavailable: {status.error ?? "unknown error"}</p>
      </div>
    );
  }

  return (
    <div className="lms-math gd-status">
      <h3>Ollama status</h3>
      <div className="gd-status-grid">
        <div>
          <span>Model</span>
          <strong>{status.model}</strong>
        </div>
        <div>
          <span>Base URL</span>
          <strong>{status.base_url}</strong>
        </div>
        <div>
          <span>Last latency</span>
          <strong>{status.last_latency_ms?.toFixed(1) ?? "-"} ms</strong>
        </div>
        <div>
          <span>Models</span>
          <strong>{status.models?.length ?? 0}</strong>
        </div>
      </div>
    </div>
  );
}
