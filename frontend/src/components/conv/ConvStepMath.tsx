import type { ConvStepResult } from "../../hooks/conv/useConvApi";

type ConvStepMathProps = {
  step: ConvStepResult | null;
};

export function ConvStepMath({ step }: ConvStepMathProps) {
  return (
    <div className="step-math-panel">
      <h3>
        {step
          ? `Step Calculation at (${step.output_row}, ${step.output_col})`
          : "Hover over activation map to see calculation"}
      </h3>
      {step && (
        <div className="step-math-content">
          <div className="math-grid-container">
            {/* Patch */}
            <div className="math-component">
              <h4>Receptive Field (Patch)</h4>
              <div className="math-grid patch-grid">
                {step.patch.map((row, r) => (
                  <div key={r} className="math-row">
                    {row.map((val, c) => (
                      <span key={c} className="math-cell">
                        {val.toFixed(0)}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            </div>

            <div className="math-op">×</div>

            {/* Kernel */}
            <div className="math-component">
              <h4>Kernel Weights</h4>
              <div className="math-grid kernel-grid">
                {step.kernel.map((row, r) => (
                  <div key={r} className="math-row">
                    {row.map((val, c) => (
                      <span key={c} className="math-cell">
                        {val.toFixed(1)}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            </div>

            <div className="math-op">=</div>

            {/* Products */}
            <div className="math-component">
              <h4>Element-wise Products</h4>
              <div className="math-grid products-grid">
                {step.products.map((row, r) => (
                  <div key={r} className="math-row">
                    {row.map((val, c) => (
                      <span key={c} className="math-cell">
                        {val.toFixed(0)}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="math-result">
            <span className="result-label">Sum:</span>
            <span className="result-value">{step.sum.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
