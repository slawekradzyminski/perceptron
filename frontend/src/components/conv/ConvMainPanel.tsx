import { ConvDigitRecognition } from "./ConvDigitRecognition";
import { ConvEducation } from "./ConvEducation";
import { ConvStepMath } from "./ConvStepMath";
import {
  ConvVisualizationPanel,
  type ReceptiveFieldBounds,
} from "./ConvVisualizationPanel";
import type { ConvState, ConvStepResult } from "../../hooks/conv/useConvApi";

type HoveredCell = { row: number; col: number } | null;

type ConvMainPanelProps = {
  state: ConvState | null;
  step: ConvStepResult | null;
  apiBase: string;
  hoveredCell: HoveredCell;
  receptiveField: ReceptiveFieldBounds | null;
  onCellHover: (cell: HoveredCell) => void;
};

export function ConvMainPanel({
  state,
  step,
  apiBase,
  hoveredCell,
  receptiveField,
  onCellHover,
}: ConvMainPanelProps) {
  return (
    <div className="conv-main">
      <ConvVisualizationPanel
        state={state}
        hoveredCell={hoveredCell}
        receptiveField={receptiveField}
        onCellHover={onCellHover}
      />

      <ConvStepMath step={step} />

      <ConvDigitRecognition apiBase={apiBase} />

      <ConvEducation />
    </div>
  );
}
