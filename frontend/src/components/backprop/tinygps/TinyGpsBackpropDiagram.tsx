import type { TinyGpsStep } from "../../types";

type TinyGpsBackpropDiagramProps = {
  step: TinyGpsStep | null;
};

const fmt = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined ? "—" : value.toFixed(digits);

export function TinyGpsBackpropDiagram({ step }: TinyGpsBackpropDiagramProps) {
  const m1 = step?.params_before.m?.[0];
  const m2 = step?.params_before.m?.[1];
  const b1 = step?.params_before.b?.[0];
  const b2 = step?.params_before.b?.[1];
  const h1 = step?.logits?.[0];
  const h2 = step?.logits?.[1];
  const yhat1 = step?.probs?.[0];
  const yhat2 = step?.probs?.[1];
  const loss = step?.loss;
  const dl_dm1 = step?.grads.m?.[0];
  const dl_dm2 = step?.grads.m?.[1];
  const dl_db1 = step?.grads.b?.[0];
  const dl_db2 = step?.grads.b?.[1];
  const active = Boolean(step);

  return (
    <div className="panel backprop-card tinygps-diagram tinygps-diagram-wide">
      <h3>Backprop Flow</h3>
      <svg viewBox="0 0 520 190" role="img" aria-label="Backprop diagram">
        <rect x={0} y={0} width={520} height={190} rx={14} fill="#fffdf8" />
        <circle cx={70} cy={95} r={24} fill="#fff" stroke="#cbb8a4" strokeWidth={2} />
        <text x={70} y={99} textAnchor="middle" fontSize={14} fill="#6a5a4a">
          x
        </text>

        <circle cx={160} cy={60} r={18} fill="#fff" stroke="#cbb8a4" strokeWidth={2} />
        <circle cx={160} cy={130} r={18} fill="#fff" stroke="#cbb8a4" strokeWidth={2} />
        <text x={160} y={64} textAnchor="middle" fontSize={12} fill="#6a5a4a">
          b1
        </text>
        <text x={160} y={134} textAnchor="middle" fontSize={12} fill="#6a5a4a">
          b2
        </text>

        <line x1={94} y1={88} x2={142} y2={70} stroke="#cbb8a4" strokeWidth={2} />
        <line x1={94} y1={102} x2={142} y2={120} stroke="#cbb8a4" strokeWidth={2} />
        <g>
          <title>m1 update: m1 ← m1 - lr * ∂L/∂m1</title>
          <text x={106} y={78} fontSize={11} fill="#6a5a4a">
            m1={fmt(m1)}
          </text>
        </g>
        <g>
          <title>m2 update: m2 ← m2 - lr * ∂L/∂m2</title>
          <text x={106} y={126} fontSize={11} fill="#6a5a4a">
            m2={fmt(m2)}
          </text>
        </g>

        <rect x={240} y={40} width={90} height={110} rx={12} fill="#fff" stroke="#cbb8a4" strokeWidth={2} />
        <g>
          <title>h1 = m1 * x + b1</title>
          <text x={285} y={78} textAnchor="middle" fontSize={11} fill="#6a5a4a">
            h1={fmt(h1)}
          </text>
        </g>
        <g>
          <title>h2 = m2 * x + b2</title>
          <text x={285} y={106} textAnchor="middle" fontSize={11} fill="#6a5a4a">
            h2={fmt(h2)}
          </text>
        </g>
        <g>
          <title>softmax([h1,h2]) → ŷ</title>
          <text x={285} y={132} textAnchor="middle" fontSize={10} fill="#6a5a4a">
            softmax
          </text>
        </g>

        <line x1={178} y1={70} x2={240} y2={70} stroke="#cbb8a4" strokeWidth={2} />
        <line x1={178} y1={120} x2={240} y2={120} stroke="#cbb8a4" strokeWidth={2} />

        <line x1={330} y1={80} x2={400} y2={80} stroke="#cbb8a4" strokeWidth={2} />
        <line x1={330} y1={120} x2={400} y2={120} stroke="#cbb8a4" strokeWidth={2} />
        <g>
          <title>ŷ1 = exp(h1) / (exp(h1)+exp(h2))</title>
          <text x={410} y={84} fontSize={11} fill="#6a5a4a">
            ŷ1={fmt(yhat1)}
          </text>
        </g>
        <g>
          <title>ŷ2 = exp(h2) / (exp(h1)+exp(h2))</title>
          <text x={410} y={124} fontSize={11} fill="#6a5a4a">
            ŷ2={fmt(yhat2)}
          </text>
        </g>

        <path
          d="M400 80 Q420 100 400 120"
          stroke="#e26b3c"
          strokeWidth={2}
          fill="none"
          className={active ? "tinygps-backprop-pulse" : ""}
        />
        <text x={388} y={106} fontSize={10} fill="#e26b3c">
          ∂L={fmt(loss)}
        </text>

        <g>
          <title>b1 update: b1 ← b1 - lr * ∂L/∂b1</title>
          <text x={160} y={24} textAnchor="middle" fontSize={10} fill="#6a5a4a">
            b1={fmt(b1)}
          </text>
        </g>
        <g>
          <title>b2 update: b2 ← b2 - lr * ∂L/∂b2</title>
          <text x={160} y={176} textAnchor="middle" fontSize={10} fill="#6a5a4a">
            b2={fmt(b2)}
          </text>
        </g>

        <g>
          <title>∂L/∂m1 = (ŷ1 - y1) * x</title>
          <text x={20} y={20} fontSize={9} fill="#6a5a4a">
            ∂L/∂m1={fmt(dl_dm1, 2)}
          </text>
        </g>
        <g>
          <title>∂L/∂m2 = (ŷ2 - y2) * x</title>
          <text x={20} y={178} fontSize={9} fill="#6a5a4a">
            ∂L/∂m2={fmt(dl_dm2, 2)}
          </text>
        </g>
        <g>
          <title>∂L/∂b1 = (ŷ1 - y1)</title>
          <text x={250} y={20} fontSize={9} fill="#6a5a4a">
            ∂L/∂b1={fmt(dl_db1, 2)}
          </text>
        </g>
        <g>
          <title>∂L/∂b2 = (ŷ2 - y2)</title>
          <text x={250} y={178} fontSize={9} fill="#6a5a4a">
            ∂L/∂b2={fmt(dl_db2, 2)}
          </text>
        </g>

        <title>
          h1 = m1*x + b1
          {"\n"}h2 = m2*x + b2
          {"\n"}ŷ = softmax([h1,h2])
          {"\n"}∂L/∂m = (ŷ - y) * x
          {"\n"}∂L/∂b = (ŷ - y)
        </title>
      </svg>
      <p className="diag-note">Values update each step; gradients flow backward from loss.</p>
    </div>
  );
}
