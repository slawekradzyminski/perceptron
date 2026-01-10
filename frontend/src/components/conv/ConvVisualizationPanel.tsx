import { ConvActivationGrid } from "./ConvActivationGrid";
import { ConvImageGrid } from "./ConvImageGrid";
import type { ConvState } from "../../hooks/conv/useConvApi";

type HoveredCell = { row: number; col: number } | null;

export type ReceptiveFieldBounds = {
  startRow: number;
  startCol: number;
  endRow: number;
  endCol: number;
};

type ConvVisualizationPanelProps = {
  state: ConvState | null;
  hoveredCell: HoveredCell;
  receptiveField: ReceptiveFieldBounds | null;
  onCellHover: (cell: HoveredCell) => void;
};

export function ConvVisualizationPanel({
  state,
  hoveredCell,
  receptiveField,
  onCellHover,
}: ConvVisualizationPanelProps) {
  return (
    <div className="conv-visualization">
      {state && (
        <ConvImageGrid
          image={state.image}
          imageSize={state.image_size}
          imageName={state.image_name}
          inputShape={state.input_shape}
          receptiveField={receptiveField}
        />
      )}

      <div className="conv-arrow">
        <span className="arrow-symbol">→</span>
        <span className="arrow-label">Convolve</span>
      </div>

      {state && (
        <ConvActivationGrid
          activationMap={state.activation_map}
          activationMapNormalized={state.activation_map_normalized}
          outputShape={state.output_shape}
          hoveredCell={hoveredCell}
          onCellHover={onCellHover}
        />
      )}
    </div>
  );
}
