export type Point = { x: number; y: number; label: 1 | -1 };
export type Grid = number[][];

type GridKind = "input" | "weights-before" | "weights-after" | "contrib";

export type State = {
  datasetName: string;
  sampleIdx: number;
  points: Point[];
  w: [number, number];
  b: number;
  lastStep: {
    x: [number, number];
    y: number;
    score: number;
    pred: number;
    mistake: boolean;
    deltaW: [number, number];
    deltaB: number;
    lr: number;
  } | null;
  nextInput: { x: [number, number]; y: number } | null;
  apiError: string | null;
  grids: {
    input: Grid;
    weightsBefore: Grid;
    weightsAfter: Grid;
    contrib: Grid;
  };
  biasBefore: number;
};

export type AppRefs = {
  plot: HTMLCanvasElement;
  input: HTMLCanvasElement;
  weightsBefore: HTMLCanvasElement;
  weightsAfter: HTMLCanvasElement;
  contrib: HTMLCanvasElement;
  scoreEl: HTMLElement;
  lrEl: HTMLInputElement;
  lrValueEl: HTMLElement;
  stepBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  datasetEl: HTMLSelectElement;
  apiBase: HTMLInputElement;
  calcX: HTMLElement;
  calcY: HTMLElement;
  calcW: HTMLElement;
  calcB: HTMLElement;
  calcS: HTMLElement;
  calcPred: HTMLElement;
  calcScoreEq: HTMLElement;
  calcUpdateEq: HTMLElement;
  calcNote: HTMLElement;
  stateW: HTMLElement;
  stateB: HTMLElement;
  stateDs: HTMLElement;
  stateIdx: HTMLElement;
  nextX: HTMLElement;
  nextY: HTMLElement;
  lastStepSection: HTMLElement;
  tooltip: HTMLElement;
  biasBeforeEl: HTMLElement;
  biasAfterEl: HTMLElement;
  biasBeforeCard: HTMLElement;
  biasAfterCard: HTMLElement;
};

function makeGrid(size: number, fill: (r: number, c: number) => number): Grid {
  return Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) => fill(r, c)),
  );
}

function weightsToGrid(w: [number, number]): Grid {
  return [[w[0], w[1]]];
}

function inputToGrid(x: [number, number]): Grid {
  return [[x[0], x[1]]];
}

function contributionGrid(a: Grid, b: Grid): Grid {
  return a.map((row, r) => row.map((v, c) => v * b[r][c]));
}

function valueColor(v: number): string {
  if (v === 0) return "#ffffff";
  return v > 0 ? "#c7f9cc" : "#f6b6b0";
}

function setBiasCardColor(card: HTMLElement, value: number) {
  card.style.backgroundColor = valueColor(value);
}

export function createInitialState(): State {
  const initialInput: [number, number] = [-1, -1];
  const weightGrid = weightsToGrid([0, 0]);
  const inputGrid = inputToGrid(initialInput);
  const contribGrid = contributionGrid(inputGrid, weightGrid);

  return {
    datasetName: "or",
    sampleIdx: 0,
    points: [
      { x: -1, y: -1, label: -1 },
      { x: -1, y: 1, label: 1 },
      { x: 1, y: -1, label: 1 },
      { x: 1, y: 1, label: 1 },
    ],
    w: [0, 0],
    b: 0,
    lastStep: null,
    nextInput: null,
    apiError: null,
    grids: {
      input: inputGrid,
      weightsBefore: weightGrid,
      weightsAfter: weightGrid,
      contrib: contribGrid,
    },
    biasBefore: 0,
  };
}

export function getRefs(root: Document = document): AppRefs {
  const plot = root.getElementById("plot") as HTMLCanvasElement | null;
  const input = root.getElementById("input") as HTMLCanvasElement | null;
  const weightsBefore = root.getElementById("weights-before") as HTMLCanvasElement | null;
  const weightsAfter = root.getElementById("weights-after") as HTMLCanvasElement | null;
  const contrib = root.getElementById("contrib") as HTMLCanvasElement | null;
  const scoreEl = root.getElementById("score") as HTMLElement | null;
  const lrEl = root.getElementById("lr") as HTMLInputElement | null;
  const lrValueEl = root.getElementById("lr-value") as HTMLElement | null;
  const stepBtn = root.getElementById("step") as HTMLButtonElement | null;
  const resetBtn = root.getElementById("reset") as HTMLButtonElement | null;
  const datasetEl = root.getElementById("dataset") as HTMLSelectElement | null;
  const apiBase = root.getElementById("api-base") as HTMLInputElement | null;
  const calcX = root.getElementById("calc-x") as HTMLElement | null;
  const calcY = root.getElementById("calc-y") as HTMLElement | null;
  const calcW = root.getElementById("calc-w") as HTMLElement | null;
  const calcB = root.getElementById("calc-b") as HTMLElement | null;
  const calcS = root.getElementById("calc-s") as HTMLElement | null;
  const calcPred = root.getElementById("calc-pred") as HTMLElement | null;
  const calcScoreEq = root.getElementById("calc-score-eq") as HTMLElement | null;
  const calcUpdateEq = root.getElementById("calc-update-eq") as HTMLElement | null;
  const calcNote = root.getElementById("calc-note") as HTMLElement | null;
  const stateW = root.getElementById("state-w") as HTMLElement | null;
  const stateB = root.getElementById("state-b") as HTMLElement | null;
  const stateDs = root.getElementById("state-ds") as HTMLElement | null;
  const stateIdx = root.getElementById("state-idx") as HTMLElement | null;
  const nextX = root.getElementById("next-x") as HTMLElement | null;
  const nextY = root.getElementById("next-y") as HTMLElement | null;
  const lastStepSection = root.getElementById("last-step-section") as HTMLElement | null;
  const tooltip = root.getElementById("cell-tooltip") as HTMLElement | null;
  const biasBeforeEl = root.getElementById("bias-before") as HTMLElement | null;
  const biasAfterEl = root.getElementById("bias-after") as HTMLElement | null;
  const biasBeforeCard = root.getElementById("bias-before-card") as HTMLElement | null;
  const biasAfterCard = root.getElementById("bias-after-card") as HTMLElement | null;

  if (
    !plot ||
    !input ||
    !weightsBefore ||
    !weightsAfter ||
    !contrib ||
    !scoreEl ||
    !lrEl ||
    !lrValueEl ||
    !stepBtn ||
    !resetBtn ||
    !datasetEl ||
    !apiBase ||
    !calcX ||
    !calcY ||
    !calcW ||
    !calcB ||
    !calcS ||
    !calcPred ||
    !calcScoreEq ||
    !calcUpdateEq ||
    !calcNote ||
    !stateW ||
    !stateB ||
    !stateDs ||
    !stateIdx ||
    !nextX ||
    !nextY ||
    !lastStepSection ||
    !tooltip ||
    !biasBeforeEl ||
    !biasAfterEl ||
    !biasBeforeCard ||
    !biasAfterCard
  ) {
    throw new Error("Missing required DOM elements");
  }

  return {
    plot,
    input,
    weightsBefore,
    weightsAfter,
    contrib,
    scoreEl,
    lrEl,
    lrValueEl,
    stepBtn,
    resetBtn,
    datasetEl,
    apiBase,
    calcX,
    calcY,
    calcW,
    calcB,
    calcS,
    calcPred,
    calcScoreEq,
    calcUpdateEq,
    calcNote,
    stateW,
    stateB,
    stateDs,
    stateIdx,
    nextX,
    nextY,
    lastStepSection,
    tooltip,
    biasBeforeEl,
    biasAfterEl,
    biasBeforeCard,
    biasAfterCard,
  };
}

function drawBoundary(ctx: CanvasRenderingContext2D, w: [number, number], b: number) {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;

  ctx.fillStyle = "#fff9f0";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#e9dac6";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const x = (i * width) / 4;
    const y = (i * height) / 4;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#c9b79f";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  const x1 = -1.5;
  const x2 = 1.5;
  const y1 = -(w[0] * x1 + b) / w[1];
  const y2 = -(w[0] * x2 + b) / w[1];
  const mapX = (x: number) => ((x + 1.5) / 3) * width;
  const mapY = (y: number) => height - ((y + 1.5) / 3) * height;

  ctx.strokeStyle = "#1c1b19";
  ctx.lineWidth = 2.5;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(mapX(x1), mapY(y1));
  ctx.lineTo(mapX(x2), mapY(y2));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#c9b79f";
  ctx.lineWidth = 2;
  ctx.strokeRect(4, 4, width - 8, height - 8);
}

function drawPoints(ctx: CanvasRenderingContext2D, points: Point[]) {
  const mapX = (x: number) => ((x + 1.5) / 3) * ctx.canvas.width;
  const mapY = (y: number) => ctx.canvas.height - ((y + 1.5) / 3) * ctx.canvas.height;
  points.forEach((p) => {
    ctx.fillStyle = p.label === 1 ? "#3d5af1" : "#ff6a3d";
    ctx.beginPath();
    ctx.arc(mapX(p.x), mapY(p.y), 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fef5ea";
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

function drawGrid(canvas: HTMLCanvasElement, grid: Grid, palette: (v: number) => string) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const rows = grid.length;
  const cols = grid[0].length;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = palette(grid[r][c]);
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }
  ctx.strokeStyle = "#d1bda7";
  ctx.lineWidth = 3;
  for (let r = 0; r <= rows; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * cellH);
    ctx.lineTo(canvas.width, r * cellH);
    ctx.stroke();
  }
  for (let c = 0; c <= cols; c++) {
    ctx.beginPath();
    ctx.moveTo(c * cellW, 0);
    ctx.lineTo(c * cellW, canvas.height);
    ctx.stroke();
  }
}

function formatVec(v: [number, number]) {
  return `[${v[0].toFixed(2)}, ${v[1].toFixed(2)}]`;
}

function formatSign(val: number) {
  return val >= 0 ? `+${val.toFixed(2)}` : `${val.toFixed(2)}`;
}

function setTooltip(refs: AppRefs, text: string, x: number, y: number) {
  refs.tooltip.textContent = text;
  refs.tooltip.style.left = `${x + 12}px`;
  refs.tooltip.style.top = `${y + 12}px`;
  refs.tooltip.classList.remove("is-hidden");
}

function hideTooltip(refs: AppRefs) {
  refs.tooltip.classList.add("is-hidden");
}

function bindTooltip(
  refs: AppRefs,
  canvas: HTMLCanvasElement,
  kind: GridKind,
  getValue: (row: number, col: number) => string,
) {
  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const cols = 2;
    const x = event.clientX - rect.left;
    const col = Math.min(cols - 1, Math.max(0, Math.floor((x / rect.width) * cols)));
    const value = getValue(0, col);
    setTooltip(refs, `${kind}: ${value}`, event.clientX, event.clientY);
  });
  canvas.addEventListener("mouseleave", () => hideTooltip(refs));
}

function render(refs: AppRefs, state: State) {
  const ctx = refs.plot.getContext("2d");
  if (!ctx) return;
  drawBoundary(ctx, state.w, state.b);
  drawPoints(ctx, state.points);

  const sample = state.lastStep ? state.lastStep.x : (state.nextInput?.x ?? [-1, -1]);
  const inputGrid = inputToGrid(sample as [number, number]);
  const weightsAfter = weightsToGrid(state.w);
  const weightsBefore = state.lastStep
    ? weightsToGrid([state.w[0] - state.lastStep.deltaW[0], state.w[1] - state.lastStep.deltaW[1]])
    : weightsAfter;
  const biasBefore = state.lastStep ? state.b - state.lastStep.deltaB : state.b;
  const contribGrid = contributionGrid(inputGrid, weightsBefore);

  state.grids = { input: inputGrid, weightsBefore, weightsAfter, contrib: contribGrid };
  state.biasBefore = biasBefore;

  drawGrid(refs.input, inputGrid, valueColor);
  drawGrid(refs.weightsBefore, weightsBefore, valueColor);
  drawGrid(refs.weightsAfter, weightsAfter, valueColor);
  drawGrid(refs.contrib, contribGrid, valueColor);

  const displayScore = state.lastStep ? state.lastStep.score : (contribGrid.flat().reduce((acc, v) => acc + v, 0) + biasBefore);
  refs.scoreEl.textContent = displayScore.toFixed(2);

  refs.biasBeforeEl.textContent = biasBefore.toFixed(2);
  refs.biasAfterEl.textContent = state.b.toFixed(2);
  setBiasCardColor(refs.biasBeforeCard, biasBefore);
  setBiasCardColor(refs.biasAfterCard, state.b);

  refs.stateW.textContent = formatVec(state.w);
  refs.stateB.textContent = state.b.toFixed(2);
  refs.stateDs.textContent = state.datasetName;
  refs.stateIdx.textContent = state.sampleIdx.toString();

  if (state.nextInput) {
    refs.nextX.textContent = `[${state.nextInput.x[0]}, ${state.nextInput.x[1]}]`;
    refs.nextY.textContent = state.nextInput.y > 0 ? "+1" : "-1";
  }

  if (state.lastStep) {
    refs.lastStepSection.classList.remove("is-hidden");
    const { x, y, score: s, pred, mistake, deltaW, deltaB, lr } = state.lastStep;
    const wBefore: [number, number] = [state.w[0] - deltaW[0], state.w[1] - deltaW[1]];
    const bBefore = state.b - deltaB;
    refs.calcX.textContent = `[${x[0]}, ${x[1]}]`;
    refs.calcY.textContent = y > 0 ? "+1" : "-1";
    refs.calcW.textContent = formatVec(state.w);
    refs.calcB.textContent = state.b.toFixed(2);
    refs.calcS.textContent = s.toFixed(2);
    refs.calcPred.textContent = pred > 0 ? "+1" : "-1";

    const term1 = (wBefore[0] * x[0]).toFixed(2);
    const term2 = (wBefore[1] * x[1]).toFixed(2);
    refs.calcScoreEq.textContent = `s = (${wBefore[0].toFixed(2)} * ${x[0]}) + (${wBefore[1].toFixed(2)} * ${x[1]}) + ${formatSign(bBefore)} = ${term1} + ${term2} + ${formatSign(bBefore)} = ${s.toFixed(2)}`;

    refs.calcUpdateEq.textContent = `w_before + Δw = ${formatVec(wBefore)} + [${deltaW[0].toFixed(2)}, ${deltaW[1].toFixed(2)}] -> ${formatVec(state.w)},  b_before + Δb = ${bBefore.toFixed(2)} + ${deltaB.toFixed(2)} -> ${state.b.toFixed(2)}`;
    refs.calcNote.textContent = mistake
      ? `Mistake: y * s = ${(y * s).toFixed(2)} <= 0, update applied. Score uses pre-update weights.`
      : `Correct: y * s = ${(y * s).toFixed(2)} > 0, no update. Score uses pre-update weights.`;
  } else {
    refs.lastStepSection.classList.add("is-hidden");
    refs.calcNote.textContent = state.apiError ?? "No update yet.";
  }
}

function validateStepPayload(data: any): data is {
  w: [number, number];
  b: number;
  x: [number, number];
  y: number;
  score: number;
  pred: number;
  mistake: boolean;
  delta_w: [number, number];
  delta_b: number;
  lr: number;
} {
  return (
    Array.isArray(data?.w) &&
    data.w.length === 2 &&
    Array.isArray(data?.x) &&
    data.x.length === 2 &&
    typeof data?.b === "number" &&
    typeof data?.y === "number" &&
    typeof data?.score === "number" &&
    typeof data?.pred === "number" &&
    typeof data?.mistake === "boolean" &&
    Array.isArray(data?.delta_w) &&
    data.delta_w.length === 2 &&
    typeof data?.delta_b === "number"
  );
}

async function apiStep(refs: AppRefs, state: State) {
  try {
    const res = await fetch(`${refs.apiBase.value}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: state.datasetName, lr: Number(refs.lrEl.value) }),
    });
    if (!res.ok) {
      state.apiError = `API error: ${res.status}`;
      return;
    }
    const data = await res.json();
    if (!validateStepPayload(data)) {
      state.apiError = "API response missing step details. Restart backend.";
      return;
    }
    state.apiError = null;
    state.w = data.w;
    state.b = data.b;
    state.sampleIdx = data.idx ?? state.sampleIdx;
    state.lastStep = {
      x: data.x,
      y: data.y,
      score: data.score,
      pred: data.pred,
      mistake: data.mistake,
      deltaW: data.delta_w,
      deltaB: data.delta_b,
      lr: data.lr ?? Number(refs.lrEl.value),
    };
    state.nextInput = data.next_x && data.next_y ? { x: data.next_x, y: data.next_y } : state.nextInput;
  } catch (err) {
    state.apiError = "API unreachable. Check backend.";
  }
}

async function apiReset(refs: AppRefs, state: State) {
  try {
    const res = await fetch(`${refs.apiBase.value}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: state.datasetName, lr: Number(refs.lrEl.value) }),
    });
    if (!res.ok) {
      state.apiError = `API error: ${res.status}`;
      return;
    }
    const data = await res.json();
    state.apiError = null;
    state.w = data.w;
    state.b = data.b;
    state.sampleIdx = data.idx ?? 0;
    state.nextInput = data.next_x && data.next_y ? { x: data.next_x, y: data.next_y } : null;
    state.lastStep = null;
  } catch (err) {
    state.apiError = "API unreachable. Check backend.";
  }
}

export function initApp(root: Document = document) {
  const state = createInitialState();
  const refs = getRefs(root);

  const updateLrDisplay = () => {
    refs.lrValueEl.textContent = Number(refs.lrEl.value).toFixed(1);
  };

  const step = async () => {
    await apiStep(refs, state);
    render(refs, state);
  };

  const reset = async () => {
    await apiReset(refs, state);
    render(refs, state);
  };

  refs.stepBtn.addEventListener("click", () => void step());
  refs.resetBtn.addEventListener("click", () => void reset());
  refs.lrEl.addEventListener("input", updateLrDisplay);
  refs.datasetEl.addEventListener("change", () => {
    state.datasetName = refs.datasetEl.value;
    state.points = state.datasetName === "xor"
      ? [
          { x: -1, y: -1, label: -1 },
          { x: -1, y: 1, label: 1 },
          { x: 1, y: -1, label: 1 },
          { x: 1, y: 1, label: -1 },
        ]
      : [
          { x: -1, y: -1, label: -1 },
          { x: -1, y: 1, label: 1 },
          { x: 1, y: -1, label: 1 },
          { x: 1, y: 1, label: 1 },
        ];
    state.sampleIdx = 0;
    void reset();
  });
  document.addEventListener("keydown", (event) => {
    const target = event.target as HTMLElement | null;
    const tag = target?.tagName?.toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") {
      return;
    }
    const key = event.key.toLowerCase();
    if (key === "s") {
      void step();
    }
    if (key === "r") {
      void reset();
    }
  });

  bindTooltip(refs, refs.input, "input", (row, col) => {
    const v = state.grids.input[row][col];
    return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  });
  bindTooltip(refs, refs.weightsBefore, "weights(before)", (row, col) => {
    const v = state.grids.weightsBefore[row][col];
    return `${v.toFixed(2)}`;
  });
  bindTooltip(refs, refs.weightsAfter, "weights(after)", (row, col) => {
    const v = state.grids.weightsAfter[row][col];
    let detail = `${v.toFixed(2)}`;
    if (state.lastStep) {
      const prev = v - state.lastStep.deltaW[col];
      detail = `${v.toFixed(2)} (prev ${prev.toFixed(2)}, Δ ${state.lastStep.deltaW[col].toFixed(2)})`;
    }
    return detail;
  });
  bindTooltip(refs, refs.contrib, "contrib(pre)", (row, col) => {
    const v = state.grids.contrib[row][col];
    return `${v.toFixed(2)} = x${col + 1}*w${col + 1}`;
  });

  updateLrDisplay();
  render(refs, state);
  void reset();
}
