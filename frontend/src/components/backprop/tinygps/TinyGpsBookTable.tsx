import { useMemo, useState } from "react";
import type { TinyGpsStep } from "../../types";

const formatTrim = (value: number, digits: number) => {
  const raw = value.toFixed(digits);
  return raw.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
};

type TinyGpsBookTableProps = {
  history: TinyGpsStep[];
  datasetName: string;
};

export function TinyGpsBookTable({ history, datasetName }: TinyGpsBookTableProps) {
  const [hover, setHover] = useState<{ text: string; x: number; y: number; placement: "above" | "below" } | null>(
    null,
  );

  const rows = useMemo(() => {
    return history.map((step, idx) => {
      const [h1, h2] = step.logits;
      const [yhat1, yhat2] = step.probs;
      return {
        idx,
        x: step.sample.x[0],
        m1: step.params_before.m?.[0] ?? 0,
        m2: step.params_before.m?.[1] ?? 0,
        b1: step.params_before.b?.[0] ?? 0,
        b2: step.params_before.b?.[1] ?? 0,
        dl_dm1: step.grads.m?.[0] ?? 0,
        dl_dm2: step.grads.m?.[1] ?? 0,
        dl_db1: step.grads.b?.[0] ?? 0,
        dl_db2: step.grads.b?.[1] ?? 0,
        h1,
        h2,
        yhat1,
        yhat2,
        y1: step.sample.y === 0 ? 1 : 0,
        y2: step.sample.y === 1 ? 1 : 0,
        loss: step.loss,
        acc: step.metrics_after.accuracy,
      };
    });
  }, [history]);

  if (!rows.length) {
    return (
      <div className="panel backprop-card backprop-table">
        <h3>Book Table</h3>
        <p className="diag-placeholder">Step through the exercise to fill the table.</p>
      </div>
    );
  }

  const cellHint = (row: (typeof rows)[number], key: string) => {
    switch (key) {
      case "step":
        return `Step = index in the table\nstep = ${row.idx}`;
      case "x":
        return `x = longitude for the sample\nx = ${formatTrim(row.x, 4)}`;
      case "m1":
        return `m1 (weight before update)\nm1 = ${formatTrim(row.m1, 4)}\nUpdate: m1 ← m1 - lr * ∂L/∂m1`;
      case "m2":
        return `m2 (weight before update)\nm2 = ${formatTrim(row.m2, 4)}\nUpdate: m2 ← m2 - lr * ∂L/∂m2`;
      case "dl_dm1":
        return `∂L/∂m1 = (ŷ1 - y1) * x\n= ${formatTrim(row.dl_dm1, 4)}`;
      case "dl_dm2":
        return `∂L/∂m2 = (ŷ2 - y2) * x\n= ${formatTrim(row.dl_dm2, 4)}`;
      case "b1":
        return `b1 (bias before update)\nb1 = ${formatTrim(row.b1, 4)}\nUpdate: b1 ← b1 - lr * ∂L/∂b1`;
      case "b2":
        return `b2 (bias before update)\nb2 = ${formatTrim(row.b2, 4)}\nUpdate: b2 ← b2 - lr * ∂L/∂b2`;
      case "dl_db1":
        return `∂L/∂b1 = (ŷ1 - y1)\n= ${formatTrim(row.dl_db1, 4)}`;
      case "dl_db2":
        return `∂L/∂b2 = (ŷ2 - y2)\n= ${formatTrim(row.dl_db2, 4)}`;
      case "h1":
        return `h1 = m1*x + b1\n= ${formatTrim(row.m1, 4)}*${formatTrim(row.x, 4)} + ${formatTrim(row.b1, 4)}\n= ${formatTrim(row.h1, 4)}`;
      case "h2":
        return `h2 = m2*x + b2\n= ${formatTrim(row.m2, 4)}*${formatTrim(row.x, 4)} + ${formatTrim(row.b2, 4)}\n= ${formatTrim(row.h2, 4)}`;
      case "yhat1":
        return `ŷ1 = softmax(h1,h2)\n= ${formatTrim(row.yhat1, 4)}`;
      case "yhat2":
        return `ŷ2 = softmax(h1,h2)\n= ${formatTrim(row.yhat2, 4)}`;
      case "y1":
        return `y1 = 1 if sample is Paris\n= ${row.y1}`;
      case "y2":
        return `y2 = 1 if sample is Berlin\n= ${row.y2}`;
      case "loss":
        return `Loss = -log(ŷ_correct)\n= ${formatTrim(row.loss, 4)}`;
      case "acc":
        return `Accuracy = mean over dataset\n= ${formatTrim(row.acc, 4)}`;
      default:
        return "";
    }
  };

  const handleEnter = (event: React.MouseEvent<HTMLElement>, row: (typeof rows)[number], key: string) => {
    const text = cellHint(row, key);
    if (!text) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const padding = 12;
    const tooltipHeight = 120;
    const placement = rect.top < tooltipHeight + padding ? "below" : "above";
    setHover({
      text,
      x: rect.left + rect.width / 2,
      y: placement === "above" ? rect.top - 8 : rect.bottom + 8,
      placement,
    });
  };

  return (
    <div className="panel backprop-card backprop-table">
      <h3>Book Table (TinyGPS)</h3>
      <div className="diag-note">Dataset: {datasetName}</div>
      <div className="book-table-wrap">
        <table>
          <thead>
            <tr>
              <th>step</th>
              <th>x</th>
              <th>m1</th>
              <th>m2</th>
              <th>∂l/∂m1</th>
              <th>∂l/∂m2</th>
              <th>b1</th>
              <th>b2</th>
              <th>∂l/∂b1</th>
              <th>∂l/∂b2</th>
              <th>h1</th>
              <th>h2</th>
              <th>ŷ1</th>
              <th>ŷ2</th>
              <th>y1</th>
              <th>y2</th>
              <th>loss</th>
              <th>accuracy</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.idx}
                onMouseLeave={() => setHover(null)}
              >
                <td onMouseEnter={(event) => handleEnter(event, row, "step")}>{row.idx}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "x")}>{formatTrim(row.x, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "m1")}>{formatTrim(row.m1, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "m2")}>{formatTrim(row.m2, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "dl_dm1")}>{formatTrim(row.dl_dm1, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "dl_dm2")}>{formatTrim(row.dl_dm2, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "b1")}>{formatTrim(row.b1, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "b2")}>{formatTrim(row.b2, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "dl_db1")}>{formatTrim(row.dl_db1, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "dl_db2")}>{formatTrim(row.dl_db2, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "h1")}>{formatTrim(row.h1, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "h2")}>{formatTrim(row.h2, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "yhat1")}>{formatTrim(row.yhat1, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "yhat2")}>{formatTrim(row.yhat2, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "y1")}>{row.y1}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "y2")}>{row.y2}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "loss")}>{formatTrim(row.loss, 3)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "acc")}>{formatTrim(row.acc, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hover && (
        <div
          className={`book-table-hover book-table-hover-${hover.placement}`}
          style={{ left: hover.x, top: hover.y }}
        >
          <pre>{hover.text}</pre>
        </div>
      )}
    </div>
  );
}
