import { useMemo, useState } from "react";

type LossMetric = "cross-entropy" | "l1";

type ComputedRow = {
  context: string;
  correctToken: string;
  pCorrect: number;
  topToken: string;
  topProb: number;
  loss: number;
};

type TokenLossRow = {
  context: string;
  correct_token: string;
  p_correct: number;
  top_token: string;
  top_prob: number;
  l1_loss: number;
  ce_loss: number;
};

type TokenLossExample = {
  id: string;
  title: string;
  description: string;
  average: { l1: number; ce: number };
  rows: TokenLossRow[];
};

function formatProb(value: number) {
  if (value === 0) return "0";
  if (value < 0.001) return value.toExponential(2);
  return value.toFixed(6);
}

function computeLoss(row: TokenLossRow, metric: LossMetric) {
  return metric === "l1" ? row.l1_loss : row.ce_loss;
}

type TokenLossTableCardProps = {
  examples: TokenLossExample[];
};

export function TokenLossTableCard({ examples }: TokenLossTableCardProps) {
  const [exampleId, setExampleId] = useState(examples[0]?.id ?? "");
  const [metric, setMetric] = useState<LossMetric>("cross-entropy");
  const example = useMemo(
    () => examples.find((item) => item.id === exampleId) ?? examples[0],
    [examples, exampleId],
  );
  const rows = useMemo<ComputedRow[]>(
    () =>
      example.rows.map((row) => ({
        context: row.context,
        correctToken: row.correct_token,
        pCorrect: row.p_correct,
        topToken: row.top_token,
        topProb: row.top_prob,
        loss: computeLoss(row, metric),
      })),
    [example.rows, metric],
  );
  const losses = rows.map((row) => row.loss);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const avgLoss = losses.reduce((acc, val) => acc + val, 0) / losses.length;

  return (
    <div className="lms-math lms-token-loss">
      <div className="lms-token-head">
        <div>
          <h3>Token loss table</h3>
          <p>{example.description}</p>
        </div>
        <label className="lms-token-select">
          Example
          <select value={exampleId} onChange={(event) => setExampleId(event.target.value)}>
            {examples.map((item) => (
              <option key={item.id} value={item.id}>
                {item.title}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="lms-token-controls" role="group" aria-label="Loss metric">
        <button
          type="button"
          className={metric === "cross-entropy" ? "active" : ""}
          onClick={() => setMetric("cross-entropy")}
        >
          Cross-entropy
        </button>
        <button
          type="button"
          className={metric === "l1" ? "active" : ""}
          onClick={() => setMetric("l1")}
        >
          L1 loss
        </button>
        <div className="lms-token-average">
          Average {metric}: <strong>{avgLoss.toFixed(4)}</strong>
        </div>
      </div>
      <div className="lms-table-wrap lms-token-table">
        <table className="lms-table">
          <thead>
            <tr>
              <th>Input text</th>
              <th>Top token</th>
              <th>P(top)</th>
              <th>Correct token</th>
              <th>P(correct)</th>
              <th>Loss</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isMin = row.loss === minLoss;
              const isMax = row.loss === maxLoss;
              return (
                <tr key={`${row.context}-${row.correctToken}`}>
                  <td>{row.context}</td>
                  <td>{row.topToken || "(n/a)"}</td>
                  <td>{row.topProb ? formatProb(row.topProb) : "-"}</td>
                  <td>{row.correctToken}</td>
                  <td>{formatProb(row.pCorrect)}</td>
                  <td className={isMax ? "bad" : isMin ? "ok" : ""}>{row.loss.toFixed(4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="lms-note">Highest loss highlights the most confident mistakes; lowest loss shows the most confident hits.</p>
    </div>
  );
}
