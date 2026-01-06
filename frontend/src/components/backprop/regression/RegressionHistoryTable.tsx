import type { RegressionStep } from "../../../types";
import { formatNum } from "../common/formatters";

type RegressionHistoryTableProps = {
  history: RegressionStep[];
};

export function RegressionHistoryTable({ history }: RegressionHistoryTableProps) {
  return (
    <div className="panel backprop-card backprop-table">
      <h3>Regression History</h3>
      {history.length ? (
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>x</th>
              <th>y</th>
              <th>ŷ</th>
              <th>Loss</th>
            </tr>
          </thead>
          <tbody>
            {history.map((row) => (
              <tr key={`${row.sample_idx}-${row.idx}`}>
                <td>{row.sample_idx + 1}</td>
                <td>{formatNum(row.sample.x, 3)}</td>
                <td>{formatNum(row.sample.y, 3)}</td>
                <td>{formatNum(row.y_hat, 4)}</td>
                <td>{formatNum(row.loss_value, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="diag-placeholder">No regression steps yet.</p>
      )}
    </div>
  );
}
