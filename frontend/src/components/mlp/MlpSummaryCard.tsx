import type { MlpTrainerSnapshot } from "../../types";
import { formatVec } from "../../utils/visuals";

type MlpSummaryCardProps = {
  snapshot: MlpTrainerSnapshot;
  correct: number;
  total: number;
};

export function MlpSummaryCard({ snapshot, correct, total }: MlpSummaryCardProps) {
  return (
    <div className="mlp-summary">
      <div>
        <span>Dataset</span>
        <strong>{snapshot.dataset.toUpperCase()}</strong>
      </div>
      <div>
        <span>Sample</span>
        <strong>{formatVec(snapshot.next_x)}</strong>
      </div>
      <div>
        <span>Target y</span>
        <strong>{snapshot.next_y}</strong>
      </div>
      <div>
        <span>Accuracy</span>
        <strong>
          {correct}/{total}
        </strong>
      </div>
    </div>
  );
}
