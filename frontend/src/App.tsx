import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  contributionGrid,
  createFilledGrid,
  drawBoundary,
  drawGrid,
  drawPoints,
  flattenGrid,
  formatSign,
  formatVec,
  makeGrid,
  Point,
  resizeGrid,
  reshapeToGrid,
  setGridCanvasSize,
  valueColor,
} from "./visuals";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

type LastStep = {
  x: number[];
  y: number;
  score: number;
  pred: number;
  mistake: boolean;
  deltaW: number[];
  deltaB: number;
  lr: number;
};

type CustomSample = { grid: number[][]; y: number };

type CustomConfig = {
  rows: number;
  cols: number;
  samples: CustomSample[];
};

type TooltipState = {
  visible: boolean;
  text: string;
  x: number;
  y: number;
};

function defaultCustomConfig(): CustomConfig {
  return {
    rows: 2,
    cols: 2,
    samples: [{ grid: createFilledGrid(2, 2, -1), y: -1 }],
  };
}

function pointsForDataset(
  dataset: string,
  dim: number,
  customConfig: CustomConfig,
): Point[] {
  if (dim !== 2) return [];
  if (dataset === "xor") {
    return [
      { x: -1, y: -1, label: -1 },
      { x: -1, y: 1, label: 1 },
      { x: 1, y: -1, label: 1 },
      { x: 1, y: 1, label: -1 },
    ];
  }
  if (dataset === "custom") {
    return customConfig.samples
      .map((sample) => {
        const flat = flattenGrid(sample.grid);
        if (flat.length !== 2) return null;
        return { x: flat[0], y: flat[1], label: sample.y as 1 | -1 };
      })
      .filter((point): point is Point => point !== null);
  }
  return [
    { x: -1, y: -1, label: -1 },
    { x: -1, y: 1, label: 1 },
    { x: 1, y: -1, label: 1 },
    { x: 1, y: 1, label: 1 },
  ];
}

function buildCustomPayload(customConfig: CustomConfig) {
  return {
    grid_rows: customConfig.rows,
    grid_cols: customConfig.cols,
    samples: customConfig.samples.map((sample) => ({
      grid: sample.grid,
      y: sample.y,
    })),
  };
}

function getCellFromEvent(event: React.MouseEvent<HTMLCanvasElement>, rows: number, cols: number) {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const col = Math.min(cols - 1, Math.max(0, Math.floor((x / rect.width) * cols)));
  const row = Math.min(rows - 1, Math.max(0, Math.floor((y / rect.height) * rows)));
  return { row, col };
}

export default function App() {
  const [datasetName, setDatasetName] = useState("or");
  const [gridRows, setGridRows] = useState(1);
  const [gridCols, setGridCols] = useState(2);
  const [sampleCount, setSampleCount] = useState(4);
  const [sampleIdx, setSampleIdx] = useState(0);
  const [w, setW] = useState<number[]>([0, 0]);
  const [b, setB] = useState(0);
  const [lastStep, setLastStep] = useState<LastStep | null>(null);
  const [nextInput, setNextInput] = useState<{ x: number[]; y: number } | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [lr, setLr] = useState(1);
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [customConfig, setCustomConfig] = useState(defaultCustomConfig());
  const [customApplied, setCustomApplied] = useState(false);
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false,
    text: "",
    x: 0,
    y: 0,
  });

  const plotRef = useRef<HTMLCanvasElement | null>(null);
  const inputRef = useRef<HTMLCanvasElement | null>(null);
  const weightsBeforeRef = useRef<HTMLCanvasElement | null>(null);
  const weightsAfterRef = useRef<HTMLCanvasElement | null>(null);
  const contribRef = useRef<HTMLCanvasElement | null>(null);

  const dim = gridRows * gridCols;
  const showBoundary = dim === 2;
  const points = useMemo(
    () => pointsForDataset(datasetName, dim, customConfig),
    [datasetName, dim, customConfig],
  );

  const fallbackInput = useMemo(() => Array.from({ length: dim }, () => -1), [dim]);
  const sample = lastStep?.x ?? nextInput?.x ?? fallbackInput;
  const inputGrid = useMemo(() => reshapeToGrid(sample, gridRows, gridCols), [sample, gridRows, gridCols]);
  const weightsAfter = useMemo(() => reshapeToGrid(w, gridRows, gridCols), [w, gridRows, gridCols]);
  const weightsBefore = useMemo(() => {
    if (!lastStep) return weightsAfter;
    return reshapeToGrid(w.map((val, i) => val - lastStep.deltaW[i]), gridRows, gridCols);
  }, [w, lastStep, gridRows, gridCols, weightsAfter]);
  const biasBefore = lastStep ? b - lastStep.deltaB : b;
  const contribGrid = useMemo(
    () => contributionGrid(inputGrid, weightsBefore),
    [inputGrid, weightsBefore],
  );

  const displayScore = lastStep
    ? lastStep.score
    : contribGrid.flat().reduce((acc, v) => acc + v, 0) + biasBefore;

  const apiReset = useCallback(
    async (forceCustomPayload = false) => {
      try {
        const payload: Record<string, unknown> = {
          dataset: datasetName,
          lr,
        };
        if (datasetName === "custom" && forceCustomPayload) {
          Object.assign(payload, buildCustomPayload(customConfig));
        }
        const res = await fetch(`${apiBase}/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          setApiError(`API error: ${res.status}`);
          return;
        }
        const data = await res.json();
        setApiError(null);
        setDatasetName(data.dataset ?? datasetName);
        setW(data.w ?? w);
        setB(data.b ?? b);
        setSampleIdx(data.idx ?? 0);
        setGridRows(data.grid_rows ?? gridRows);
        setGridCols(data.grid_cols ?? gridCols);
        setSampleCount(data.sample_count ?? sampleCount);
        setNextInput(data.next_x && data.next_y ? { x: data.next_x, y: data.next_y } : null);
        setLastStep(null);
        setCustomApplied(data.dataset === "custom");
      } catch {
        setApiError("API unreachable. Check backend.");
      }
    },
    [apiBase, datasetName, lr, customConfig, w, b, gridRows, gridCols, sampleCount],
  );

  const apiStep = useCallback(async () => {
    try {
      if (datasetName === "custom" && !customApplied) {
        setApiError("Apply the custom dataset before stepping.");
        return;
      }
      const res = await fetch(`${apiBase}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: datasetName, lr }),
      });
      if (!res.ok) {
        setApiError(`API error: ${res.status}`);
        return;
      }
      const data = await res.json();
      if (!Array.isArray(data?.w) || !Array.isArray(data?.x)) {
        setApiError("API response missing step details. Restart backend.");
        return;
      }
      setApiError(null);
      setDatasetName(data.dataset ?? datasetName);
      setW(data.w);
      setB(data.b);
      setSampleIdx(data.idx ?? sampleIdx);
      setGridRows(data.grid_rows ?? gridRows);
      setGridCols(data.grid_cols ?? gridCols);
      setSampleCount(data.sample_count ?? sampleCount);
      setLastStep({
        x: data.x,
        y: data.y,
        score: data.score,
        pred: data.pred,
        mistake: data.mistake,
        deltaW: data.delta_w,
        deltaB: data.delta_b,
        lr: data.lr ?? lr,
      });
      setNextInput(data.next_x && data.next_y ? { x: data.next_x, y: data.next_y } : nextInput);
    } catch {
      setApiError("API unreachable. Check backend.");
    }
  }, [apiBase, datasetName, lr, customApplied, gridRows, gridCols, sampleIdx, sampleCount, nextInput]);

  useEffect(() => {
    void apiReset();
  }, []);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      const key = event.key.toLowerCase();
      if (key === "s") void apiStep();
      if (key === "r") void apiReset();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [apiStep, apiReset]);

  useEffect(() => {
    if (inputRef.current) setGridCanvasSize(inputRef.current, gridRows, gridCols);
    if (weightsBeforeRef.current) setGridCanvasSize(weightsBeforeRef.current, gridRows, gridCols);
    if (weightsAfterRef.current) setGridCanvasSize(weightsAfterRef.current, gridRows, gridCols);
    if (contribRef.current) setGridCanvasSize(contribRef.current, gridRows, gridCols);
  }, [gridRows, gridCols]);

  useEffect(() => {
    if (inputRef.current) drawGrid(inputRef.current, inputGrid, valueColor);
    if (weightsBeforeRef.current) drawGrid(weightsBeforeRef.current, weightsBefore, valueColor);
    if (weightsAfterRef.current) drawGrid(weightsAfterRef.current, weightsAfter, valueColor);
    if (contribRef.current) drawGrid(contribRef.current, contribGrid, valueColor);
  }, [inputGrid, weightsBefore, weightsAfter, contribGrid]);

  useEffect(() => {
    if (!showBoundary) return;
    const ctx = plotRef.current?.getContext("2d");
    if (!ctx) return;
    drawBoundary(ctx, w, b);
    drawPoints(ctx, points);
  }, [showBoundary, w, b, points]);

  const onDatasetChange = (value: string) => {
    setDatasetName(value);
    if (value === "custom") {
      setGridRows(customConfig.rows);
      setGridCols(customConfig.cols);
      setCustomApplied(false);
      return;
    }
    void apiReset();
  };

  const updateCustomRows = (rows: number) => {
    setCustomApplied(false);
    setCustomConfig((prev) => ({
      ...prev,
      rows,
      samples: prev.samples.map((sample) => ({
        ...sample,
        grid: resizeGrid(sample.grid, rows, prev.cols, -1),
      })),
    }));
    if (datasetName === "custom") setGridRows(rows);
  };

  const updateCustomCols = (cols: number) => {
    setCustomApplied(false);
    setCustomConfig((prev) => ({
      ...prev,
      cols,
      samples: prev.samples.map((sample) => ({
        ...sample,
        grid: resizeGrid(sample.grid, prev.rows, cols, -1),
      })),
    }));
    if (datasetName === "custom") setGridCols(cols);
  };

  const addCustomSample = () => {
    setCustomApplied(false);
    setCustomConfig((prev) => ({
      ...prev,
      samples: [
        ...prev.samples,
        { grid: createFilledGrid(prev.rows, prev.cols, -1), y: 1 },
      ],
    }));
  };

  const removeCustomSample = (index: number) => {
    setCustomApplied(false);
    setCustomConfig((prev) => ({
      ...prev,
      samples: prev.samples.filter((_, idx) => idx !== index),
    }));
  };

  const toggleCustomCell = (sampleIndex: number, row: number, col: number) => {
    setCustomApplied(false);
    setCustomConfig((prev) => ({
      ...prev,
      samples: prev.samples.map((sample, idx) => {
        if (idx !== sampleIndex) return sample;
        return {
          ...sample,
          grid: makeGrid(prev.rows, prev.cols, (r, c) => {
            if (r !== row || c !== col) return sample.grid[r][c];
            return sample.grid[r][c] === 1 ? -1 : 1;
          }),
        };
      }),
    }));
  };

  const updateSampleLabel = (index: number, value: number) => {
    setCustomApplied(false);
    setCustomConfig((prev) => ({
      ...prev,
      samples: prev.samples.map((sample, idx) =>
        idx === index ? { ...sample, y: value } : sample,
      ),
    }));
  };

  const applyCustomDataset = () => {
    setDatasetName("custom");
    setGridRows(customConfig.rows);
    setGridCols(customConfig.cols);
    setSampleIdx(0);
    void apiReset(true);
  };

  const handleCanvasMove = (
    grid: number[][],
    label: string,
    getValue: (row: number, col: number) => string,
  ) =>
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const { row, col } = getCellFromEvent(event, grid.length, grid[0].length);
      setTooltip({
        visible: true,
        text: `${label}: ${getValue(row, col)}`,
        x: event.clientX + 12,
        y: event.clientY + 12,
      });
    };

  return (
    <div id="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Perceptron Visual Lab</p>
          <h1>Make the decision boundary move.</h1>
          <p className="subhead">A hands-on view of weights, mistakes, and linearly separable data.</p>
        </div>
        <div className="controls">
          <label>
            <span>Dataset</span>
            <select value={datasetName} onChange={(event) => onDatasetChange(event.target.value)}>
              <option value="or">OR</option>
              <option value="xor">XOR</option>
              <option value="custom">Custom</option>
            </select>
          </label>
          <div className="static-field">
            <span>Switchboard</span>
            <strong>{gridRows}×{gridCols}</strong>
          </div>
          <button onClick={() => void apiStep()}>Step</button>
          <button onClick={() => void apiReset()}>Reset</button>
          <label>
            <span>Learning rate</span>
            <div className="lr-row">
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.1"
                value={lr}
                onChange={(event) => setLr(Number(event.target.value))}
              />
              <span>{lr.toFixed(1)}</span>
            </div>
          </label>
          <label>
            <span>API base</span>
            <input
              type="text"
              value={apiBase}
              onChange={(event) => setApiBase(event.target.value)}
            />
          </label>
          <p className="hotkeys">Shortcuts: <strong>S</strong> step, <strong>R</strong> reset</p>
        </div>
      </header>

      <main className="grid">
        {showBoundary && (
          <section className="panel" id="boundary-panel">
            <h2>2D Boundary</h2>
            <canvas ref={plotRef} id="plot" className="plot-canvas" width={420} height={420} />
          </section>
        )}

        <section className="panel">
          <h2>Switchboard</h2>
          <div className="switchboard">
            <div className="switch-row">
              <div>
                <h3>Input</h3>
                <canvas
                  ref={inputRef}
                  onMouseMove={handleCanvasMove(inputGrid, "input", (row, col) => {
                    const v = inputGrid[row][col];
                    return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
                  })}
                  onMouseLeave={() => setTooltip((prev) => ({ ...prev, visible: false }))}
                />
              </div>
              <div>
                <h3>Weights (before)</h3>
                <canvas
                  ref={weightsBeforeRef}
                  onMouseMove={handleCanvasMove(weightsBefore, "weights(before)", (row, col) => {
                    const v = weightsBefore[row][col];
                    return v.toFixed(2);
                  })}
                  onMouseLeave={() => setTooltip((prev) => ({ ...prev, visible: false }))}
                />
              </div>
            </div>
            <div className="switch-row">
              <div>
                <h3>Contribution (pre-update)</h3>
                <canvas
                  ref={contribRef}
                  onMouseMove={handleCanvasMove(contribGrid, "contrib(pre)", (row, col) => {
                    const v = contribGrid[row][col];
                    return `${v.toFixed(2)} = x${row + 1}${col + 1}*w${row + 1}${col + 1}`;
                  })}
                  onMouseLeave={() => setTooltip((prev) => ({ ...prev, visible: false }))}
                />
              </div>
              <div>
                <h3>Bias (before)</h3>
                <div
                  className="bias-card"
                  style={{ backgroundColor: valueColor(biasBefore), aspectRatio: `${gridCols} / ${gridRows}` }}
                >
                  <div className="bias-value">{biasBefore.toFixed(2)}</div>
                  <div className="bias-note">Adds to score</div>
                </div>
              </div>
            </div>
          </div>
          {tooltip.visible && (
            <div
              className="cell-tooltip"
              style={{ left: tooltip.x, top: tooltip.y }}
            >
              {tooltip.text}
            </div>
          )}
          <div className="meter">
            <span>Score (s)</span>
            <strong data-testid="score-value">{displayScore.toFixed(2)}</strong>
          </div>
        </section>

        <section className="panel">
          <h2>After update</h2>
          <div className="switchboard">
            <div className="switch-row">
              <div>
                <h3>Weights (after)</h3>
                <canvas
                  ref={weightsAfterRef}
                  onMouseMove={handleCanvasMove(weightsAfter, "weights(after)", (row, col) => {
                    const v = weightsAfter[row][col];
                    let detail = v.toFixed(2);
                    if (lastStep) {
                      const index = row * gridCols + col;
                      const prev = v - lastStep.deltaW[index];
                      detail = `${v.toFixed(2)} (prev ${prev.toFixed(2)}, Δ ${lastStep.deltaW[index].toFixed(2)})`;
                    }
                    return detail;
                  })}
                  onMouseLeave={() => setTooltip((prev) => ({ ...prev, visible: false }))}
                />
              </div>
              <div>
                <h3>Bias (after)</h3>
                <div
                  className="bias-card bias-after"
                  style={{ backgroundColor: valueColor(b), aspectRatio: `${gridCols} / ${gridRows}` }}
                >
                  <div className="bias-value">{b.toFixed(2)}</div>
                  <div className="bias-note">After update</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="panel calc-panel">
          <h2>Step math</h2>
          <div className="calc-section">
            <h3>Current state</h3>
            <div className="calc-grid">
              <div><span>Weights w</span><strong>{formatVec(w)}</strong></div>
              <div><span>Bias b</span><strong>{b.toFixed(2)}</strong></div>
              <div><span>Dataset</span><strong>{datasetName}</strong></div>
              <div><span>Sample idx</span><strong>{sampleIdx}</strong></div>
            </div>
          </div>
          <div className="calc-section">
            <h3>Next input (what we expect)</h3>
            <div className="calc-grid">
              <div><span>Input x</span><strong>{formatVec(nextInput?.x ?? fallbackInput)}</strong></div>
              <div><span>Label y</span><strong>{nextInput?.y === 1 ? "+1" : "-1"}</strong></div>
            </div>
          </div>
          {lastStep && (
            <div className="calc-section">
              <h3>Last step calculation</h3>
              <div className="calc-grid">
                <div><span>Input x</span><strong>{formatVec(lastStep.x)}</strong></div>
                <div><span>Label y</span><strong>{lastStep.y === 1 ? "+1" : "-1"}</strong></div>
                <div><span>Weights (after)</span><strong>{formatVec(w)}</strong></div>
                <div><span>Bias (after)</span><strong>{b.toFixed(2)}</strong></div>
                <div><span>Score s</span><strong>{lastStep.score.toFixed(2)}</strong></div>
                <div><span>Prediction ŷ</span><strong>{lastStep.pred === 1 ? "+1" : "-1"}</strong></div>
              </div>
              <p className="calc-eq">
                {(() => {
                  const wBefore = w.map((val, i) => val - lastStep.deltaW[i]);
                  const dot = wBefore.reduce((acc, val, i) => acc + val * lastStep.x[i], 0);
                  return `s = w·x + b = ${dot.toFixed(2)} + ${formatSign(b - lastStep.deltaB)} = ${lastStep.score.toFixed(2)}`;
                })()}
              </p>
              <p className="calc-eq">
                {(() => {
                  const wBefore = w.map((val, i) => val - lastStep.deltaW[i]);
                  return `w_before + Δw = ${formatVec(wBefore)} + ${formatVec(lastStep.deltaW)} -> ${formatVec(w)},  b_before + Δb = ${(b - lastStep.deltaB).toFixed(2)} + ${lastStep.deltaB.toFixed(2)} -> ${b.toFixed(2)}`;
                })()}
              </p>
              <p className="calc-note">
                {lastStep.mistake
                  ? `Mistake: y * s = ${(lastStep.y * lastStep.score).toFixed(2)} <= 0, update applied. Score uses pre-update weights.`
                  : `Correct: y * s = ${(lastStep.y * lastStep.score).toFixed(2)} > 0, no update. Score uses pre-update weights.`}
              </p>
            </div>
          )}
          {!lastStep && apiError && (
            <div className="calc-section">
              <p className="calc-note">{apiError}</p>
            </div>
          )}
        </section>

        <section className="panel explain">
          <h2>What happens on each step</h2>
          <div className="math">
            <p><strong>Score:</strong> <span>s = w · x + b</span></p>
            <p><strong>Prediction:</strong> <span>ŷ = sign(s)</span></p>
            <p><strong>Mistake?</strong> <span>y · s ≤ 0</span></p>
            <p><strong>Update (only if mistake):</strong> <span>w ← w + η y x, b ← b + η y</span></p>
          </div>
          <p className="detail">
            The input grid shows the switches (x). The weight grid shows the dials (w). The contribution grid
            shows each cell’s product xᵢ·wᵢ before the sum. The score meter is the sum of all contributions plus bias.
          </p>
          <p className="detail">
            For XOR, no single line separates the classes, so the boundary keeps moving without converging.
          </p>
        </section>

        {datasetName === "custom" && (
          <section className="panel custom-panel">
            <h2>Custom dataset</h2>
            <div className="custom-controls">
              <label>
                <span>Rows</span>
                <select
                  value={customConfig.rows}
                  onChange={(event) => updateCustomRows(Number(event.target.value))}
                >
                  {[1, 2, 3, 4, 5].map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
              </label>
              <label>
                <span>Cols</span>
                <select
                  value={customConfig.cols}
                  onChange={(event) => updateCustomCols(Number(event.target.value))}
                >
                  {[1, 2, 3, 4, 5].map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
              </label>
              <button type="button" onClick={addCustomSample}>Add sample</button>
              <button type="button" onClick={applyCustomDataset}>Apply custom dataset</button>
            </div>
            <div className="custom-samples">
              {customConfig.samples.map((sample, index) => (
                <div className="sample-card" key={`sample-${index}`}>
                  <div className="sample-head">
                    <span>Sample {index + 1}</span>
                    <div className="sample-actions">
                      <select
                        value={sample.y}
                        onChange={(event) => updateSampleLabel(index, Number(event.target.value))}
                      >
                        <option value={1}>+1</option>
                        <option value={-1}>-1</option>
                      </select>
                      <button
                        type="button"
                        onClick={() => removeCustomSample(index)}
                        disabled={customConfig.samples.length <= 1}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                  <div
                    className="sample-grid"
                    style={{ gridTemplateColumns: `repeat(${customConfig.cols}, 1fr)` }}
                  >
                    {sample.grid.map((row, r) =>
                      row.map((cell, c) => (
                        <button
                          key={`cell-${index}-${r}-${c}`}
                          type="button"
                          className="sample-cell"
                          style={{ backgroundColor: valueColor(cell) }}
                          onClick={() => toggleCustomCell(index, r, c)}
                          title={cell.toString()}
                        />
                      )),
                    )}
                  </div>
                </div>
              ))}
            </div>
            <p className="custom-hint">
              Click cells to toggle between -1 and +1. Labels define the expected output.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}
