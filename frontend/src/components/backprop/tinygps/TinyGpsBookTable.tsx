import { useMemo, useState, useRef } from "react";
import type { TinyGpsStep } from "../../../types";

const formatTrim = (value: number, digits: number) => {
  const raw = value.toFixed(digits);
  return raw.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
};

type TinyGpsBookTableProps = {
  history: TinyGpsStep[];
  datasetName: string;
};

type RowData = {
  idx: number;
  x: number | [number, number]; // longitude or [lon, lat] for 2D
  m: number[]; // weights per class (1D mode)
  b: number[]; // biases per class
  dl_dm: number[]; // gradients for weights
  dl_db: number[]; // gradients for biases
  h: number[]; // logits per class
  yhat: number[]; // probabilities per class
  y: number[]; // one-hot ground truth
  loss: number;
  acc: number;
  mode: "1d" | "2d";
};

export function TinyGpsBookTable({ history, datasetName }: TinyGpsBookTableProps) {
  const [hover, setHover] = useState<{ text: string; left: number; top: number } | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Determine number of classes from first history entry
  const numClasses = history.length > 0 ? history[0].cities.length : 2;
  const mode = history.length > 0 ? history[0].mode : "1d";
  const cities = history.length > 0 ? history[0].cities : [];

  const rows = useMemo<RowData[]>(() => {
    return history.map((step, idx) => {
      // Build one-hot y vector
      const y = Array(step.cities.length).fill(0);
      y[step.sample.y] = 1;

      return {
        idx,
        x: step.mode === "2d" ? [step.sample.x[0], step.sample.x[1]] : step.sample.x[0],
        m: step.params_before.m ?? [],
        b: step.params_before.b ?? [],
        dl_dm: step.grads.m ?? [],
        dl_db: step.grads.b ?? [],
        h: step.logits,
        yhat: step.probs,
        y,
        loss: step.loss,
        acc: step.metrics_after.accuracy,
        mode: step.mode,
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

  // Dynamic cell hint based on class index
  const cellHint = (row: RowData, key: string) => {
    // Handle indexed keys like m_0, dl_dm_1, etc.
    const matchM = key.match(/^m_(\d+)$/);
    const matchDlDm = key.match(/^dl_dm_(\d+)$/);
    const matchB = key.match(/^b_(\d+)$/);
    const matchDlDb = key.match(/^dl_db_(\d+)$/);
    const matchH = key.match(/^h_(\d+)$/);
    const matchYhat = key.match(/^yhat_(\d+)$/);
    const matchY = key.match(/^y_(\d+)$/);

    if (matchM) {
      const i = parseInt(matchM[1]);
      const cityName = cities[i] || `class ${i + 1}`;
      return `m${i + 1} (weight for ${cityName})\nm${i + 1} = ${formatTrim(row.m[i] ?? 0, 4)}\nUpdate: m${i + 1} ← m${i + 1} - lr * ∂L/∂m${i + 1}`;
    }
    if (matchDlDm) {
      const i = parseInt(matchDlDm[1]);
      const xStr = typeof row.x === "number" ? formatTrim(row.x, 4) : `[${formatTrim(row.x[0], 4)}, ${formatTrim(row.x[1], 4)}]`;
      return `∂L/∂m${i + 1} = (ŷ${i + 1} - y${i + 1}) * x\n= ${formatTrim(row.dl_dm[i] ?? 0, 4)}\nx = ${xStr}`;
    }
    if (matchB) {
      const i = parseInt(matchB[1]);
      const cityName = cities[i] || `class ${i + 1}`;
      return `b${i + 1} (bias for ${cityName})\nb${i + 1} = ${formatTrim(row.b[i] ?? 0, 4)}\nUpdate: b${i + 1} ← b${i + 1} - lr * ∂L/∂b${i + 1}`;
    }
    if (matchDlDb) {
      const i = parseInt(matchDlDb[1]);
      return `∂L/∂b${i + 1} = (ŷ${i + 1} - y${i + 1})\n= ${formatTrim(row.dl_db[i] ?? 0, 4)}`;
    }
    if (matchH) {
      const i = parseInt(matchH[1]);
      const xVal = typeof row.x === "number" ? row.x : row.x[0];
      if (row.mode === "1d") {
        return `h${i + 1} = m${i + 1}*x + b${i + 1}\n= ${formatTrim(row.m[i] ?? 0, 4)}*${formatTrim(xVal, 4)} + ${formatTrim(row.b[i] ?? 0, 4)}\n= ${formatTrim(row.h[i] ?? 0, 4)}`;
      } else {
        return `h${i + 1} = M${i + 1}·x + b${i + 1}\n= ${formatTrim(row.h[i] ?? 0, 4)} (2D mode)`;
      }
    }
    if (matchYhat) {
      const i = parseInt(matchYhat[1]);
      const hList = row.h.map((_, j) => `h${j + 1}`).join(",");
      return `ŷ${i + 1} = softmax(${hList})\n= ${formatTrim(row.yhat[i] ?? 0, 4)}`;
    }
    if (matchY) {
      const i = parseInt(matchY[1]);
      const cityName = cities[i] || `class ${i + 1}`;
      return `y${i + 1} = 1 if sample is ${cityName}\n= ${row.y[i]}`;
    }

    switch (key) {
      case "step":
        return `Step = index in the table\nstep = ${row.idx}`;
      case "x":
        if (typeof row.x === "number") {
          return `x = longitude for the sample\nx = ${formatTrim(row.x, 4)}`;
        } else {
          return `x = [lon, lat] for the sample\nx = [${formatTrim(row.x[0], 4)}, ${formatTrim(row.x[1], 4)}]`;
        }
      case "x1":
        return `x₁ = normalized longitude\n(lon - Paris center lon)\nx₁ = ${formatTrim(typeof row.x === "number" ? row.x : row.x[0], 4)}`;
      case "x2":
        return `x₂ = normalized latitude\n(lat - Paris center lat)\nx₂ = ${formatTrim(typeof row.x === "number" ? 0 : row.x[1], 4)}`;
      case "loss":
        return `Loss = -log(ŷ_correct)\n= ${formatTrim(row.loss, 4)}`;
      case "acc":
        return `Accuracy = mean over dataset\n= ${formatTrim(row.acc, 4)}`;
      default:
        return "";
    }
  };

  const handleEnter = (event: React.MouseEvent<HTMLElement>, row: RowData, key: string) => {
    const text = cellHint(row, key);
    if (!text || !wrapperRef.current) return;
    const cellRect = event.currentTarget.getBoundingClientRect();
    const wrapperRect = wrapperRef.current.getBoundingClientRect();
    // Position relative to wrapper
    const left = cellRect.left - wrapperRect.left + cellRect.width / 2;
    const top = cellRect.top - wrapperRect.top - 8;
    setHover({ text, left, top });
  };

  // Generate array indices for dynamic columns
  const classIndices = Array.from({ length: numClasses }, (_, i) => i);

  // Compact mode for 3+ classes or 2D mode
  const isCompact = numClasses >= 3 || mode === "2d";
  const tableClassName = `tinygps-book-table ${isCompact ? "tinygps-book-table--compact" : ""}`;

  // Get x1 and x2 values for 2D mode
  const getX1 = (x: number | [number, number]) => (typeof x === "number" ? x : x[0]);
  const getX2 = (x: number | [number, number]) => (typeof x === "number" ? 0 : x[1]);

  return (
    <div className="panel backprop-card backprop-table tinygps-book-table-container">
      <h3>Book Table (TinyGPS)</h3>
      <div className="diag-note">
        Dataset: {datasetName} ({numClasses} classes, {mode} mode)
      </div>
      <div className="book-table-wrap" ref={wrapperRef} style={{ position: "relative" }}>
        <table className={tableClassName}>
          <thead>
            <tr>
              <th>#</th>
              {mode === "2d" ? (
                <>
                  <th>x₁</th>
                  <th>x₂</th>
                </>
              ) : (
                <th>x</th>
              )}
              {/* m columns (only for 1D mode) */}
              {mode === "1d" && classIndices.map((i) => <th key={`m${i}`}>m{i + 1}</th>)}
              {mode === "1d" && classIndices.map((i) => <th key={`dl_dm${i}`}>∂m{i + 1}</th>)}
              {/* b columns */}
              {classIndices.map((i) => <th key={`b${i}`}>b{i + 1}</th>)}
              {classIndices.map((i) => <th key={`dl_db${i}`}>∂b{i + 1}</th>)}
              {/* h, ŷ, y columns */}
              {classIndices.map((i) => <th key={`h${i}`}>h{i + 1}</th>)}
              {classIndices.map((i) => <th key={`yhat${i}`}>ŷ{i + 1}</th>)}
              {classIndices.map((i) => <th key={`y${i}`}>y{i + 1}</th>)}
              <th>L</th>
              <th>acc</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.idx} onMouseLeave={() => setHover(null)}>
                <td onMouseEnter={(event) => handleEnter(event, row, "step")}>{row.idx}</td>
                {mode === "2d" ? (
                  <>
                    <td onMouseEnter={(event) => handleEnter(event, row, "x1")}>{formatTrim(getX1(row.x), 2)}</td>
                    <td onMouseEnter={(event) => handleEnter(event, row, "x2")}>{formatTrim(getX2(row.x), 2)}</td>
                  </>
                ) : (
                  <td onMouseEnter={(event) => handleEnter(event, row, "x")}>{formatTrim(getX1(row.x), 3)}</td>
                )}
                {/* m values (only for 1D mode) */}
                {mode === "1d" &&
                  classIndices.map((i) => (
                    <td key={`m${i}`} onMouseEnter={(event) => handleEnter(event, row, `m_${i}`)}>
                      {formatTrim(row.m[i] ?? 0, 2)}
                    </td>
                  ))}
                {mode === "1d" &&
                  classIndices.map((i) => (
                    <td key={`dl_dm${i}`} onMouseEnter={(event) => handleEnter(event, row, `dl_dm_${i}`)}>
                      {formatTrim(row.dl_dm[i] ?? 0, 2)}
                    </td>
                  ))}
                {/* b values */}
                {classIndices.map((i) => (
                  <td key={`b${i}`} onMouseEnter={(event) => handleEnter(event, row, `b_${i}`)}>
                    {formatTrim(row.b[i] ?? 0, 2)}
                  </td>
                ))}
                {classIndices.map((i) => (
                  <td key={`dl_db${i}`} onMouseEnter={(event) => handleEnter(event, row, `dl_db_${i}`)}>
                    {formatTrim(row.dl_db[i] ?? 0, 2)}
                  </td>
                ))}
                {/* h, ŷ, y values */}
                {classIndices.map((i) => (
                  <td key={`h${i}`} onMouseEnter={(event) => handleEnter(event, row, `h_${i}`)}>
                    {formatTrim(row.h[i] ?? 0, 2)}
                  </td>
                ))}
                {classIndices.map((i) => (
                  <td key={`yhat${i}`} onMouseEnter={(event) => handleEnter(event, row, `yhat_${i}`)}>
                    {formatTrim(row.yhat[i] ?? 0, 2)}
                  </td>
                ))}
                {classIndices.map((i) => (
                  <td key={`y${i}`} onMouseEnter={(event) => handleEnter(event, row, `y_${i}`)}>
                    {row.y[i]}
                  </td>
                ))}
                <td onMouseEnter={(event) => handleEnter(event, row, "loss")}>{formatTrim(row.loss, 2)}</td>
                <td onMouseEnter={(event) => handleEnter(event, row, "acc")}>{formatTrim(row.acc, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {hover && (
          <div
            className="tinygps-book-tooltip"
            style={{ left: hover.left, top: hover.top }}
          >
            <pre>{hover.text}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
