import type { MlpTrainerSnapshot } from "../../types";

export function MlpWeightList({ snapshot }: { snapshot: MlpTrainerSnapshot }) {
  const w1 = snapshot.hidden.weights[0] ?? [];
  const w2 = snapshot.hidden.weights[1] ?? [];
  return (
    <div className="mlp-weights">
      <h3>Weights</h3>
      <div className="mlp-weight-grid">
        <div className="mlp-weight-row">
          <span>w₁₁¹</span>
          <strong>{w1[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₁₂¹</span>
          <strong>{w1[1]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₂₁¹</span>
          <strong>{w2[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₂₂¹</span>
          <strong>{w2[1]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>b₁¹</span>
          <strong>{snapshot.hidden.bias[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>b₂¹</span>
          <strong>{snapshot.hidden.bias[1]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₁²</span>
          <strong>{snapshot.output.weights[0]?.[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>w₂²</span>
          <strong>{snapshot.output.weights[0]?.[1]?.toFixed(2) ?? "0.00"}</strong>
        </div>
        <div className="mlp-weight-row">
          <span>b¹²</span>
          <strong>{snapshot.output.bias[0]?.toFixed(2) ?? "0.00"}</strong>
        </div>
      </div>
    </div>
  );
}
