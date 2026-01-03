import type { MlpTrainerSnapshot } from "../../types";

export function MlpNetworkDiagram({ snapshot }: { snapshot: MlpTrainerSnapshot }) {
  const w1 = snapshot.hidden.weights[0] ?? [0, 0];
  const w2 = snapshot.hidden.weights[1] ?? [0, 0];
  const b1 = snapshot.hidden.bias[0] ?? 0;
  const b2 = snapshot.hidden.bias[1] ?? 0;
  const wOut = snapshot.output.weights[0] ?? [0, 0];
  const bOut = snapshot.output.bias[0] ?? 0;

  return (
    <div className="mlp-diagram">
      <h3>Network diagram</h3>
      <svg viewBox="0 0 520 260" role="img" aria-label="MLP diagram">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="#5a4733" />
          </marker>
        </defs>
        <circle cx="60" cy="60" r="26" className="mlp-node input" />
        <text x="60" y="65" textAnchor="middle">x₁</text>
        <circle cx="60" cy="160" r="26" className="mlp-node input" />
        <text x="60" y="165" textAnchor="middle">x₂</text>

        <circle cx="220" cy="60" r="30" className="mlp-node hidden" />
        <text x="220" y="65" textAnchor="middle">h₁</text>
        <circle cx="220" cy="160" r="30" className="mlp-node hidden" />
        <text x="220" y="165" textAnchor="middle">h₂</text>

        <circle cx="380" cy="110" r="32" className="mlp-node output" />
        <text x="380" y="115" textAnchor="middle">ŷ</text>

        <line x1="86" y1="60" x2="190" y2="60" markerEnd="url(#arrow)" />
        <line x1="86" y1="60" x2="190" y2="160" markerEnd="url(#arrow)" />
        <line x1="86" y1="160" x2="190" y2="60" markerEnd="url(#arrow)" />
        <line x1="86" y1="160" x2="190" y2="160" markerEnd="url(#arrow)" />

        <line x1="250" y1="60" x2="348" y2="110" markerEnd="url(#arrow)" />
        <line x1="250" y1="160" x2="348" y2="110" markerEnd="url(#arrow)" />

        <text x="125" y="45" className="mlp-weight-text">w₁₁¹={w1[0].toFixed(2)}</text>
        <text x="135" y="140" className="mlp-weight-text">w₁₂¹={w1[1].toFixed(2)}</text>
        <text x="120" y="110" className="mlp-weight-text">w₂₁¹={w2[0].toFixed(2)}</text>
        <text x="125" y="200" className="mlp-weight-text">w₂₂¹={w2[1].toFixed(2)}</text>

        <text x="270" y="45" className="mlp-weight-text">w₁²={wOut[0].toFixed(2)}</text>
        <text x="270" y="175" className="mlp-weight-text">w₂²={wOut[1].toFixed(2)}</text>

        <text x="210" y="20" className="mlp-bias-text">b₁¹={b1.toFixed(2)}</text>
        <text x="210" y="230" className="mlp-bias-text">b₂¹={b2.toFixed(2)}</text>
        <text x="380" y="40" className="mlp-bias-text">b¹²={bOut.toFixed(2)}</text>
      </svg>
    </div>
  );
}
