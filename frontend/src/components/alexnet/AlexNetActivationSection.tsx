import type { ActivationGrid } from "../../hooks/alexnet/useAlexNetApi";

type AlexNetActivationSectionProps = {
  activations: ActivationGrid | null;
  selectedSample: string;
  customImageUrl: string | null;
  hoveredFilter: number | null;
};

export function AlexNetActivationSection({
  activations,
  selectedSample,
  customImageUrl,
  hoveredFilter,
}: AlexNetActivationSectionProps) {
  if (!activations) return null;

  return (
    <div className="viz-section">
      <h3>
        Activation Maps
        <span className="hint"> — Where each filter fires on the input</span>
      </h3>

      <div className="activation-header">
        <div className="input-reference">
          <span className="ref-label">Analyzing:</span>
          <img
            src={
              customImageUrl || `data:image/png;base64,${activations.input_image}`
            }
            alt="Input being analyzed"
            className="ref-image"
          />
          <span className="ref-name">
            {selectedSample === "custom" ? "Uploaded image" : selectedSample}
          </span>
        </div>
      </div>

      <div className="activation-grid">
        {activations.activations.map((act) => (
          <div
            key={act.index}
            className={`activation-cell ${
              hoveredFilter === act.index ? "highlighted" : ""
            }`}
            title={`Filter ${act.index} — max: ${act.max_activation.toFixed(2)}`}
          >
            <img
              src={`data:image/png;base64,${act.image}`}
              alt={`Activation ${act.index}`}
            />
            <span className="activation-index">{act.index}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
