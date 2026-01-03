import type { MlpTrainerSnapshot } from "../../types";
import { formatVec } from "../../utils/visuals";

function MlpEvalRow({ row }: { row: MlpTrainerSnapshot["evals"][number] }) {
  const predLabel = row.pred === 1 ? "+1" : "-1";
  return (
    <div className="mlp-eval-row">
      <span>x={formatVec(row.x)}</span>
      <span>y={row.y}</span>
      <span>p̂={row.p_hat.toFixed(3)}</span>
      <span>ŷ={predLabel}</span>
    </div>
  );
}

export function MlpEvalList({ evals }: { evals: MlpTrainerSnapshot["evals"] }) {
  return (
    <div className="mlp-eval">
      <h3>Predictions on dataset</h3>
      <div className="mlp-eval-grid">
        {evals.map((row, idx) => (
          <MlpEvalRow key={idx} row={row} />
        ))}
      </div>
    </div>
  );
}
