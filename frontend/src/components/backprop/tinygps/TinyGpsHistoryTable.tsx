import type { TinyGpsStep } from "../../../types";
import { formatNum, formatVec } from "../common/formatters";

type TinyGpsHistoryTableProps = {
  history: TinyGpsStep[];
};

export function TinyGpsHistoryTable({ history }: TinyGpsHistoryTableProps) {
  return (
    <div className="panel backprop-card backprop-table">
      <h3>TinyGPS History</h3>
      {history.length ? (
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>City</th>
              <th>x</th>
              <th>Loss</th>
              <th>Acc</th>
            </tr>
          </thead>
          <tbody>
            {history.map((row) => (
              <tr key={`${row.sample_idx}-${row.idx}`}>
                <td>{row.sample_idx + 1}</td>
                <td>{row.sample.city}</td>
                <td>{formatVec(row.sample.x, 4)}</td>
                <td>{formatNum(row.loss, 4)}</td>
                <td>{formatNum(row.metrics_after.accuracy, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="diag-placeholder">No TinyGPS steps yet.</p>
      )}
    </div>
  );
}
