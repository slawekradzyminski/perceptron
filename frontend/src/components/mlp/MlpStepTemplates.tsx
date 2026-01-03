import type { MlpInternalsResponse } from "../../types";
import { TemplateGrid } from "./TemplateGrid";

export function MlpStepTemplates({ step }: { step: MlpInternalsResponse }) {
  return (
    <div className="template-section">
      <h3>Last step templates</h3>
      <p className="diag-note">Before, gradients, and after update for each hidden unit.</p>
      <div className="template-grid">
        {step.hidden.templates_before.map((grid, idx) => (
          <TemplateGrid key={`step-before-${idx}`} label={`H${idx + 1} (before)`} grid={grid} />
        ))}
        {step.gradients.templates.map((grid, idx) => (
          <TemplateGrid key={`step-grad-${idx}`} label={`ΔH${idx + 1}`} grid={grid} />
        ))}
        {step.hidden.templates_after.map((grid, idx) => (
          <TemplateGrid key={`step-after-${idx}`} label={`H${idx + 1} (after)`} grid={grid} />
        ))}
      </div>
    </div>
  );
}
