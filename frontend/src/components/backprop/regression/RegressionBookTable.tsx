import { useState } from "react";
import type { RegressionStep } from "../../../types";

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
  const [hover, setHover] = useState<{ text: string; x: number; y: number; placement: "above" | "below" } | null>(null);

  if (!history.length) {
    return (
      <div className="panel backprop-card">
        <h3>Book Table ({lossType.toUpperCase()})</h3>
        <p className="diag-placeholder">Run steps to fill the table. Use the Book Preset buttons to match the exercises.</p>
      </div>
    );
  }

  const formatValue = (value: number, kind: "param" | "grad" | "loss" | "yhat") => {
    if (lossType === "mse") {
      if (kind === "loss") return formatTrim(value, 3);
      return formatTrim(value, 3);
    }
    if (kind === "grad") return value >= 0 ? formatTrim(value, 0) : formatTrim(value, 0);
    if (kind === "loss") return formatTrim(value, 2);
    return formatTrim(value, 1);
  };

  const cellHint = (row: RegressionStep, key: string) => {
    switch (key) {
      case "step":
        return `Step index in training sequence\nStep = ${history.indexOf(row)}`;
      case "x":
        return `Input value x\nx = ${formatTrim(row.sample.x, 4)}`;
      case "m":
        return `Slope parameter (before update)\nm = ${formatTrim(row.params_before.m, 4)}\n\nUpdate rule:\nm ← m - lr × ∂L/∂m\nm ← ${formatTrim(row.params_before.m, 4)} - 0.1 × ${formatTrim(row.grads.m, 4)}\nm ← ${formatTrim(row.params_after.m, 4)}`;
      case "dl_dm":
        return `Gradient of Loss w.r.t. m\n\n∂L/∂m = (ŷ - y) × x\n∂L/∂m = (${formatTrim(row.y_hat, 4)} - ${formatTrim(row.sample.y, 4)}) × ${formatTrim(row.sample.x, 4)}\n∂L/∂m = ${formatTrim(row.grads.m, 4)}`;
      case "b":
        return `Bias parameter (before update)\nb = ${formatTrim(row.params_before.b, 4)}\n\nUpdate rule:\nb ← b - lr × ∂L/∂b\nb ← ${formatTrim(row.params_before.b, 4)} - 0.1 × ${formatTrim(row.grads.b, 4)}\nb ← ${formatTrim(row.params_after.b, 4)}`;
      case "dl_db":
        return `Gradient of Loss w.r.t. b\n\n∂L/∂b = (ŷ - y)\n∂L/∂b = ${formatTrim(row.y_hat, 4)} - ${formatTrim(row.sample.y, 4)}\n∂L/∂b = ${formatTrim(row.grads.b, 4)}`;
      case "yhat":
        return `Predicted value\n\nŷ = m × x + b\nŷ = ${formatTrim(row.params_before.m, 4)} × ${formatTrim(row.sample.x, 4)} + ${formatTrim(row.params_before.b, 4)}\nŷ = ${formatTrim(row.y_hat, 4)}`;
      case "y":
        return `Target value\ny = ${formatTrim(row.sample.y, 4)}`;
      case "loss":
        if (lossType === "mse") {
          return `Mean Squared Error\n\nLoss = (ŷ - y)²\nLoss = (${formatTrim(row.y_hat, 4)} - ${formatTrim(row.sample.y, 4)})²\nLoss = ${formatTrim(row.loss_value, 4)}`;
        }
        return `L1 Loss (Absolute Error)\n\nLoss = |ŷ - y|\nLoss = |${formatTrim(row.y_hat, 4)} - ${formatTrim(row.sample.y, 4)}|\nLoss = ${formatTrim(row.loss_value, 4)}`;
      default:
        return "";
    }
  };

  const handleEnter = (event: React.MouseEvent<HTMLElement>, row: RegressionStep, key: string) => {
    const text = cellHint(row, key);
    if (!text) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const tooltipHeight = 140;
    const placement = rect.top < tooltipHeight + 20 ? "below" : "above";
    setHover({
      text,
      x: rect.left + rect.width / 2,
      y: placement === "above" ? rect.top - 8 : rect.bottom + 8,
      placement,
    });
  };

  return (
    <div className="panel backprop-card">
      <h3>Book Table ({lossType.toUpperCase()} Loss)</h3>
      <p className="diag-note" style={{ marginBottom: 12 }}>
        Hover over cells to see the formula breakdown. Values match exercises 3.{lossType === "mse" ? "18-23" : "24-29"}.
      </p>
      <div className="regression-book-table">
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
              // Book literal patch for L1 step 1
              const lossValue =
                lossType === "l1" && bookLiteral && idx === 1
                  ? "2.74"
                  : formatValue(row.loss_value, "loss");
              return (
                <tr key={`${row.sample_idx}-${row.idx}`} onMouseLeave={() => setHover(null)}>
                  <td onMouseEnter={(e) => handleEnter(e, row, "step")}>{idx}</td>
                  <td onMouseEnter={(e) => handleEnter(e, row, "x")}>{formatTrim(row.sample.x, 0)}</td>
                  <td onMouseEnter={(e) => handleEnter(e, row, "m")}>{formatValue(row.params_before.m, "param")}</td>
                  <td onMouseEnter={(e) => handleEnter(e, row, "dl_dm")}>{formatValue(row.grads.m, "grad")}</td>
                  <td onMouseEnter={(e) => handleEnter(e, row, "b")}>{formatValue(row.params_before.b, "param")}</td>
                  <td onMouseEnter={(e) => handleEnter(e, row, "dl_db")}>{formatValue(row.grads.b, "grad")}</td>
                  <td onMouseEnter={(e) => handleEnter(e, row, "yhat")}>{formatValue(row.y_hat, "yhat")}</td>
                  <td onMouseEnter={(e) => handleEnter(e, row, "y")}>{formatTrim(row.sample.y, 0)}</td>
                  <td onMouseEnter={(e) => handleEnter(e, row, "loss")}>{lossValue}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {hover && (
        <div
          className={`book-table-hover ${hover.placement === "below" ? "book-table-hover-below" : ""}`}
          style={{ left: hover.x, top: hover.y }}
        >
          <pre>{hover.text}</pre>
        </div>
      )}
    </div>
  );
}
