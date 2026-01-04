import type { RegressionStep } from "../../types";
import { formatNum, formatVec } from "../common/formatters";

const formatTrim = (value: number, digits: number) => {
  const raw = value.toFixed(digits);
  return raw.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
};

type RegressionBookTableProps = {
  history: RegressionStep[];
  lossType: "mse" | "l1";
  bookLiteral: boolean;
};

export function RegressionBookTable({ history, lossType, bookLiteral }: RegressionBookTableProps) {
  if (!history.length) {
    return (
      <div className="panel backprop-card backprop-table">
        <h3>Book Table</h3>
        <p className="diag-placeholder">Run steps to fill the table.</p>
      </div>
    );
  }

  const formatValue = (value: number, kind: "param" | "grad" | "loss" | "yhat") => {
    if (lossType === "mse") {
      if (kind === "loss") return formatTrim(value, 3);
      return formatTrim(value, 3);
    }
    if (kind === "grad") return formatTrim(value, 0);
    if (kind === "loss") return formatTrim(value, 2);
    return formatTrim(value, 1);
  };

  return (
    <div className="panel backprop-card backprop-table">
      <h3>Book Table</h3>
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>x</th>
            <th>m</th>
            <th>∂L/∂m</th>
            <th>b</th>
            <th>∂L/∂b</th>
            <th>ŷ</th>
            <th>y</th>
            <th>Loss</th>
          </tr>
        </thead>
        <tbody>
          {history.map((row, idx) => {
            const lossValue =
              lossType === "l1" && bookLiteral && idx === 1
                ? "2.74"
                : formatValue(row.loss_value, "loss");
            return (
              <tr key={`${row.sample_idx}-${row.idx}`}>
                <td>{idx}</td>
                <td>{formatTrim(row.sample.x, 3)}</td>
                <td>{formatValue(row.params_before.m, "param")}</td>
                <td>{formatValue(row.grads.m, "grad")}</td>
                <td>{formatValue(row.params_before.b, "param")}</td>
                <td>{formatValue(row.grads.b, "grad")}</td>
                <td>{formatValue(row.y_hat, "yhat")}</td>
                <td>{formatTrim(row.sample.y, 3)}</td>
                <td>{lossValue}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="diag-note">
        Values are computed in full precision and rounded for display. L1 Book-Literal mode patches step 1 loss to 2.74.
      </p>
    </div>
  );
}
