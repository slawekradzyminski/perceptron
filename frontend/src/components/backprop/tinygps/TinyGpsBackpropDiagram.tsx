import type { TinyGpsStep } from "../../../types";

type TinyGpsBackpropDiagramProps = {
  step: TinyGpsStep | null;
};

const fmt = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined ? "—" : value.toFixed(digits);

// Colors for different neurons/classes
const NEURON_COLORS = [
  { stroke: "#2a7b8f", fill: "#f5f9fc", text: "#2a7b8f" }, // teal
  { stroke: "#2f7d4f", fill: "#f5fcf7", text: "#2f7d4f" }, // green
  { stroke: "#7b5aa8", fill: "#f9f5fc", text: "#7b5aa8" }, // purple
  { stroke: "#b8845a", fill: "#fff8f0", text: "#b8845a" }, // orange
];

export function TinyGpsBackpropDiagram({ step }: TinyGpsBackpropDiagramProps) {
  // Don't render until first step
  if (!step) {
    return (
      <div className="panel backprop-card tinygps-diagram">
        <h3>Backprop Flow</h3>
        <p className="diag-placeholder">Step through the exercise to see the backprop flow diagram.</p>
      </div>
    );
  }

  const numClasses = step.cities?.length ?? 2;
  const mode = step.mode ?? "1d";
  const active = true; // Always active since we have a step

  const x = step?.sample?.x;
  const pred = step?.pred;
  const correct = step?.correct;
  const cities = step?.cities ?? [];
  const loss = step?.loss;

  // Extract arrays
  const m = step?.params_before.m ?? [];
  const b = step?.params_before.b ?? [];
  const h = step?.logits ?? [];
  const yhat = step?.probs ?? [];
  const dl_dm = step?.grads.m ?? [];
  const dl_db = step?.grads.b ?? [];

  // Dynamic layout calculations - increased for better visibility
  const svgWidth = 1020;
  const svgHeight = numClasses <= 2 ? 300 : numClasses === 3 ? 380 : 460;
  const neuronSpacing = numClasses <= 2 ? 140 : numClasses === 3 ? 120 : 100;
  const startY = numClasses <= 2 ? 100 : numClasses === 3 ? 90 : 80;
  const centerY = svgHeight / 2;

  // Positions for neurons
  const neuronYPositions = Array.from({ length: numClasses }, (_, i) => startY + i * neuronSpacing);

  return (
    <div className="panel backprop-card tinygps-diagram">
      <h3>Backprop Flow ({numClasses} classes, {mode})</h3>
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} role="img" aria-label="Backprop diagram" className="tinygps-diagram-svg">
        <defs>
          {/* Gradient for forward pass arrows */}
          <linearGradient id="forwardGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#3d5af1" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#3d5af1" stopOpacity="1" />
          </linearGradient>
          {/* Gradient for backward pass arrows */}
          <linearGradient id="backwardGrad" x1="100%" y1="0%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="#e26b3c" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#e26b3c" stopOpacity="1" />
          </linearGradient>
          {/* Marker for forward arrow */}
          <marker id="arrowForward" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,1 L6,4 L0,7 Z" fill="#3d5af1" />
          </marker>
          {/* Marker for backward arrow */}
          <marker id="arrowBackward" markerWidth="8" markerHeight="8" refX="2" refY="4" orient="auto">
            <path d="M6,1 L0,4 L6,7 Z" fill="#e26b3c" />
          </marker>
          {/* Glow filter for active elements */}
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <radialGradient id="inputGradient">
            <stop offset="0%" stopColor="#e8edff" />
            <stop offset="100%" stopColor="#d0d8ff" />
          </radialGradient>
        </defs>

        {/* Background */}
        <rect x={0} y={0} width={svgWidth} height={svgHeight} rx={16} fill="#fffdf8" />

        {/* Legend */}
        <g transform="translate(20, 20)">
          <text fontSize={10} fill="#7b6a57" fontWeight={600}>LEGEND</text>
          <line x1={0} y1={18} x2={30} y2={18} stroke="#3d5af1" strokeWidth={2} markerEnd="url(#arrowForward)" />
          <text x={36} y={22} fontSize={9} fill="#6a5a4a">Forward pass</text>
          <line x1={100} y1={18} x2={130} y2={18} stroke="#e26b3c" strokeWidth={2} strokeDasharray="5 3" className={active ? "tinygps-backprop-pulse" : ""} />
          <text x={136} y={22} fontSize={9} fill="#6a5a4a">Gradient flow (backprop)</text>
        </g>

        {/* INPUT NODE */}
        <g transform={`translate(90, ${centerY})`}>
          <circle r={44} fill="url(#inputGradient)" stroke="#3d5af1" strokeWidth={3} filter={active ? "url(#glow)" : ""}>
            {active && (
              <animate attributeName="stroke-opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite" />
            )}
          </circle>
          <text y={-8} textAnchor="middle" fontSize={20} fontWeight={700} fill="#3d5af1">x</text>
          <text y={16} textAnchor="middle" fontSize={mode === "2d" ? 12 : 14} fontFamily="IBM Plex Mono, monospace" fill="#1c1b19">
            {x !== undefined
              ? mode === "2d"
                ? `[${fmt(x[0], 2)}, ${fmt(x[1], 2)}]`
                : fmt(x[0], 4)
              : mode === "2d" ? "[lon, lat]" : "lon"}
          </text>
        </g>

        {/* NEURONS - Dynamic based on numClasses */}
        {neuronYPositions.map((yPos, i) => {
          const color = NEURON_COLORS[i % NEURON_COLORS.length];
          const neuronHeight = mode === "2d" ? 65 : 75;
          return (
            <g key={`neuron-${i}`} transform={`translate(250, ${yPos})`}>
              <rect x={-60} y={-neuronHeight / 2} width={120} height={neuronHeight} rx={14} fill="#fff" stroke={color.stroke} strokeWidth={2.5} />
              <text y={-neuronHeight / 2 + 16} textAnchor="middle" fontSize={11} fill={color.text} fontWeight={600}>
                NEURON {i + 1}
              </text>
              {mode === "1d" ? (
                <>
                  <text y={6} textAnchor="middle" fontSize={13} fontFamily="IBM Plex Mono, monospace" fill="#1c1b19">
                    m{i + 1}={fmt(m[i])}
                  </text>
                  <text y={24} textAnchor="middle" fontSize={13} fontFamily="IBM Plex Mono, monospace" fill="#1c1b19">
                    b{i + 1}={fmt(b[i])}
                  </text>
                </>
              ) : (
                <text y={12} textAnchor="middle" fontSize={13} fontFamily="IBM Plex Mono, monospace" fill="#1c1b19">
                  b{i + 1}={fmt(b[i])}
                </text>
              )}
            </g>
          );
        })}

        {/* Forward arrows from input to neurons */}
        {neuronYPositions.map((yPos, i) => (
          <line
            key={`fwd-input-${i}`}
            x1={134}
            y1={centerY + (yPos - centerY) * 0.2}
            x2={185}
            y2={yPos}
            stroke="#3d5af1"
            strokeWidth={2.5}
            markerEnd="url(#arrowForward)"
            className={active ? "tinygps-forward-pulse" : ""}
          />
        ))}

        {/* LOGITS - Dynamic */}
        {neuronYPositions.map((yPos, i) => {
          const color = NEURON_COLORS[i % NEURON_COLORS.length];
          return (
            <g key={`logit-${i}`} transform={`translate(420, ${yPos})`}>
              <rect x={-50} y={-28} width={100} height={56} rx={12} fill={color.fill} stroke={color.stroke} strokeWidth={2.5} />
              <text y={-6} textAnchor="middle" fontSize={11} fill={color.text} fontWeight={600}>LOGIT</text>
              <text y={16} textAnchor="middle" fontSize={14} fontFamily="IBM Plex Mono, monospace" fill="#1c1b19">
                h{i + 1}={fmt(h[i])}
              </text>
            </g>
          );
        })}

        {/* Forward arrows to logits */}
        {neuronYPositions.map((yPos, i) => (
          <line
            key={`fwd-logit-${i}`}
            x1={310}
            y1={yPos}
            x2={365}
            y2={yPos}
            stroke="#3d5af1"
            strokeWidth={2.5}
            markerEnd="url(#arrowForward)"
            className={active ? "tinygps-forward-pulse" : ""}
          />
        ))}

        {/* SOFTMAX BOX */}
        <g transform={`translate(580, ${centerY})`}>
          <rect x={-65} y={-30 - (numClasses - 2) * 18} width={130} height={60 + (numClasses - 2) * 36} rx={16} fill="#fff8f0" stroke="#d4a574" strokeWidth={3} />
          <text y={-12 - (numClasses - 2) * 18} textAnchor="middle" fontSize={12} fill="#b8845a" fontWeight={700}>SOFTMAX</text>
          <line x1={-50} y1={4 - (numClasses - 2) * 18} x2={50} y2={4 - (numClasses - 2) * 18} stroke="#e9dccb" strokeWidth={1} />
          {Array.from({ length: numClasses }, (_, i) => (
            <text key={`yhat-${i}`} y={22 - (numClasses - 2) * 18 + i * 18} textAnchor="middle" fontSize={13} fontFamily="IBM Plex Mono, monospace" fill="#1c1b19">
              ŷ{i + 1}={fmt(yhat[i], 3)}
            </text>
          ))}
        </g>

        {/* Forward arrows to softmax */}
        {neuronYPositions.map((yPos, i) => {
          const targetY = centerY + (i - (numClasses - 1) / 2) * 16;
          return (
            <path
              key={`fwd-softmax-${i}`}
              d={`M 470 ${yPos} Q 510 ${yPos} 515 ${targetY}`}
              stroke="#3d5af1"
              strokeWidth={2.5}
              fill="none"
              markerEnd="url(#arrowForward)"
              className={active ? "tinygps-forward-pulse" : ""}
            />
          );
        })}

        {/* LOSS BOX */}
        <g transform={`translate(760, ${centerY})`}>
          <rect x={-70} y={-50} width={140} height={100} rx={16} fill={active ? "#fff0e8" : "#fff"} stroke="#e26b3c" strokeWidth={3} filter={active ? "url(#glow)" : ""}>
            {active && (
              <animate attributeName="stroke" values="#e26b3c;#ff8a5c;#e26b3c" dur="1.5s" repeatCount="indefinite" />
            )}
          </rect>
          <text y={-24} textAnchor="middle" fontSize={12} fill="#e26b3c" fontWeight={700}>CROSS-ENTROPY</text>
          <text y={6} textAnchor="middle" fontSize={18} fontFamily="IBM Plex Mono, monospace" fontWeight={700} fill="#c24a2a">
            L={fmt(loss, 4)}
          </text>
          <text y={32} textAnchor="middle" fontSize={10} fill="#a08060" fontStyle="italic">
            −log(ŷ[true])
          </text>
        </g>

        {/* Forward arrow to loss */}
        <line x1={645} y1={centerY} x2={685} y2={centerY} stroke="#3d5af1" strokeWidth={2.5} markerEnd="url(#arrowForward)" className={active ? "tinygps-forward-pulse" : ""} />

        {/* PREDICTION BADGE */}
        {pred !== undefined && (
          <g transform={`translate(910, ${centerY})`}>
            <rect x={-50} y={-32} width={100} height={64} rx={12} fill={correct ? "#d4edda" : "#f8d7da"} stroke={correct ? "#28a745" : "#dc3545"} strokeWidth={2.5} />
            <text y={-8} textAnchor="middle" fontSize={11} fill={correct ? "#155724" : "#721c24"} fontWeight={600}>
              PRED
            </text>
            <text y={18} textAnchor="middle" fontSize={14} fontFamily="IBM Plex Mono, monospace" fontWeight={700} fill={correct ? "#155724" : "#721c24"}>
              {cities[pred] || `City ${pred + 1}`}
            </text>
          </g>
        )}

        {/* BACKWARD GRADIENT FLOWS */}
        {active && (
          <>
            {/* From loss back to softmax */}
            <line x1={688} y1={centerY + 18} x2={650} y2={centerY + 18} stroke="#e26b3c" strokeWidth={2.5} strokeDasharray="6 4" className="tinygps-backprop-pulse" markerEnd="url(#arrowBackward)" />

            {/* From softmax back to logits - curved paths offset below forward arrows */}
            {neuronYPositions.map((yPos, i) => {
              const softmaxY = centerY + (i - (numClasses - 1) / 2) * 16;
              return (
                <path
                  key={`back-logit-${i}`}
                  d={`M 515 ${softmaxY + 18} C 495 ${softmaxY + 40}, 485 ${yPos + 35}, 475 ${yPos + 32}`}
                  stroke="#e26b3c"
                  strokeWidth={2.5}
                  strokeDasharray="5 3"
                  fill="none"
                  className="tinygps-backprop-pulse"
                  markerEnd="url(#arrowBackward)"
                />
              );
            })}

            {/* From logits back to neurons (where m and b live) - this is where gradients stop */}
            {neuronYPositions.map((yPos, i) => (
              <line
                key={`back-neuron-${i}`}
                x1={367}
                y1={yPos + 32}
                x2={315}
                y2={yPos + 32}
                stroke="#e26b3c"
                strokeWidth={2.5}
                strokeDasharray="5 3"
                className="tinygps-backprop-pulse"
                markerEnd="url(#arrowBackward)"
              />
            ))}
          </>
        )}

        <title>
          Forward: h_i = {mode === "1d" ? "m_i*x + b_i" : "M_i·x + b_i"} → softmax → ŷ
          Backward: ∂L/∂{mode === "1d" ? "m" : "M"} = (ŷ - y) ⊗ x, ∂L/∂b = (ŷ - y)
        </title>
      </svg>
      <p className="diag-note">
        <strong>Forward pass</strong> (blue): input → {numClasses} neurons → softmax → loss.{" "}
        <strong style={{ marginLeft: 8, color: "#e26b3c" }}>Backward pass</strong> (orange dashed): gradients flow from loss back to parameters.
      </p>
      <div className="tinygps-gradients-box">
        <div className="tinygps-gradients-title">GRADIENTS</div>
        <div className="tinygps-gradients-row">
          {mode === "1d"
            ? Array.from({ length: numClasses }, (_, i) => (
                <span key={`dm${i}`} className="tinygps-grad-item">∂L/∂m{i + 1}={fmt(dl_dm[i], 3)}</span>
              ))
            : <span className="tinygps-grad-item">M gradients: (2D matrix per class)</span>
          }
        </div>
        <div className="tinygps-gradients-row">
          {Array.from({ length: numClasses }, (_, i) => (
            <span key={`db${i}`} className="tinygps-grad-item">∂L/∂b{i + 1}={fmt(dl_db[i], 3)}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
