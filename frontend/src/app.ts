export type Point = { x: number; y: number; label: 1 | -1 };
export type Grid = number[][];

export type State = {
  datasetName: string;
  sampleIdx: number;
  points: Point[];
  gridInput: Grid;
  gridSize: number;
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
  apiError: string | null;
};

export type AppRefs = {
  plot: HTMLCanvasElement;
  input: HTMLCanvasElement;
  weights: HTMLCanvasElement;
  contrib: HTMLCanvasElement;
  scoreEl: HTMLElement;
  lrEl: HTMLInputElement;
  lrValueEl: HTMLElement;
  stepBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  datasetEl: HTMLSelectElement;
  gridSizeEl: HTMLSelectElement;
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
};

function makeGrid(size: number, fill: (r: number, c: number) => number): Grid {
  return Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) => fill(r, c)),
  );
}

function makeInputGrid(size: number): Grid {
  return makeGrid(size, (r, c) => (r === c ? 1 : -1));
}

export function createInitialState(): State {
  return {
    datasetName: "or",
    sampleIdx: 0,
    points: [
      { x: -1, y: -1, label: -1 },
      { x: -1, y: 1, label: 1 },
      { x: 1, y: -1, label: 1 },
      { x: 1, y: 1, label: 1 },
    ],
    gridSize: 3,
    gridInput: makeInputGrid(3),
    w: [0, 0],
    b: 0,
    lastStep: null,
    apiError: null,
  };
}

export function getRefs(root: Document = document): AppRefs {
  const plot = root.getElementById("plot") as HTMLCanvasElement | null;
  const input = root.getElementById("input") as HTMLCanvasElement | null;
  const weights = root.getElementById("weights") as HTMLCanvasElement | null;
  const contrib = root.getElementById("contrib") as HTMLCanvasElement | null;
  const scoreEl = root.getElementById("score") as HTMLElement | null;
  const lrEl = root.getElementById("lr") as HTMLInputElement | null;
  const lrValueEl = root.getElementById("lr-value") as HTMLElement | null;
  const stepBtn = root.getElementById("step") as HTMLButtonElement | null;
  const resetBtn = root.getElementById("reset") as HTMLButtonElement | null;
  const datasetEl = root.getElementById("dataset") as HTMLSelectElement | null;
  const gridSizeEl = root.getElementById("grid-size") as HTMLSelectElement | null;
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

  if (
    !plot ||
    !input ||
    !weights ||
    !contrib ||
    !scoreEl ||
    !lrEl ||
    !lrValueEl ||
    !stepBtn ||
    !resetBtn ||
    !datasetEl ||
    !gridSizeEl ||
    !apiBase ||
    !calcX ||
    !calcY ||
    !calcW ||
    !calcB ||
    !calcS ||
    !calcPred ||
    !calcScoreEq ||
    !calcUpdateEq ||
    !calcNote
  ) {
    throw new Error("Missing required DOM elements");
  }

  return {
    plot,
    input,
    weights,
    contrib,
    scoreEl,
    lrEl,
    lrValueEl,
    stepBtn,
    resetBtn,
    datasetEl,
    gridSizeEl,
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
  };
}

function drawBoundary(ctx: CanvasRenderingContext2D, w: [number, number], b: number) {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  const x1 = -1.5;
  const x2 = 1.5;
  const y1 = -(w[0] * x1 + b) / w[1];
  const y2 = -(w[0] * x2 + b) / w[1];
  const mapX = (x: number) => ((x + 1.5) / 3) * width;
  const mapY = (y: number) => height - ((y + 1.5) / 3) * height;

  ctx.strokeStyle = "#1c1b19";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(mapX(x1), mapY(y1));
  ctx.lineTo(mapX(x2), mapY(y2));
  ctx.stroke();
}

function drawPoints(ctx: CanvasRenderingContext2D, points: Point[]) {
  const mapX = (x: number) => ((x + 1.5) / 3) * ctx.canvas.width;
  const mapY = (y: number) => ctx.canvas.height - ((y + 1.5) / 3) * ctx.canvas.height;
  points.forEach((p) => {
    ctx.fillStyle = p.label === 1 ? "#3d5af1" : "#ff6a3d";
    ctx.beginPath();
    ctx.arc(mapX(p.x), mapY(p.y), 8, 0, Math.PI * 2);
    ctx.fill();
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
  ctx.strokeStyle = "#e3d6c7";
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

function contributionGrid(a: Grid, b: Grid): Grid {
  return a.map((row, r) => row.map((v, c) => v * b[r][c]));
}

function weightsToGrid(w: [number, number], size: number): Grid {
  const values = [w[0], w[1], -w[0], -w[1]];
  return makeGrid(size, (r, c) => values[(r + c) % values.length]);
}

function formatVec(v: [number, number]) {
  return `[${v[0].toFixed(2)}, ${v[1].toFixed(2)}]`;
}

function formatSign(val: number) {
  return val >= 0 ? `+${val.toFixed(2)}` : `${val.toFixed(2)}`;
}

function render(refs: AppRefs, state: State) {
  const ctx = refs.plot.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, refs.plot.width, refs.plot.height);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, refs.plot.width, refs.plot.height);
  drawPoints(ctx, state.points);
  drawBoundary(ctx, state.w, state.b);

  const gridWeights = weightsToGrid(state.w, state.gridSize);
  drawGrid(refs.input, state.gridInput, (v) => (v > 0 ? "#ffe0c2" : "#f3f3f3"));
  drawGrid(refs.weights, gridWeights, (v) => (v > 0 ? "#cbe2ff" : "#ffd1c2"));
  const contribGrid = contributionGrid(state.gridInput, gridWeights);
  drawGrid(refs.contrib, contribGrid, (v) => (v > 0 ? "#c7f9cc" : "#ffd6d6"));

  const score = contribGrid.flat().reduce((acc, v) => acc + v, 0) + state.b;
  refs.scoreEl.textContent = score.toFixed(2);

  if (state.lastStep) {
    const { x, y, score: s, pred, mistake, deltaW, deltaB, lr } = state.lastStep;
    refs.calcX.textContent = `[${x[0]}, ${x[1]}]`;
    refs.calcY.textContent = y > 0 ? "+1" : "-1";
    refs.calcW.textContent = formatVec(state.w);
    refs.calcB.textContent = state.b.toFixed(2);
    refs.calcS.textContent = s.toFixed(2);
    refs.calcPred.textContent = pred > 0 ? "+1" : "-1";

    const term1 = (state.w[0] * x[0]).toFixed(2);
    const term2 = (state.w[1] * x[1]).toFixed(2);
    refs.calcScoreEq.textContent = `s = (${state.w[0].toFixed(2)} * ${x[0]}) + (${state.w[1].toFixed(2)} * ${x[1]}) + ${formatSign(state.b)} = ${term1} + ${term2} + ${formatSign(state.b)} = ${s.toFixed(2)}`;

    refs.calcUpdateEq.textContent = `w <- w + ${lr.toFixed(2)} * ${y} * x = w + [${deltaW[0].toFixed(2)}, ${deltaW[1].toFixed(2)}],  b <- b + ${deltaB.toFixed(2)}`;
    refs.calcNote.textContent = mistake
      ? `Mistake: y * s = ${(y * s).toFixed(2)} <= 0, update applied.`
      : `Correct: y * s = ${(y * s).toFixed(2)} > 0, no update.`;
  } else {
    refs.calcX.textContent = "[-1, 1]";
    refs.calcY.textContent = "+1";
    refs.calcW.textContent = formatVec(state.w);
    refs.calcB.textContent = state.b.toFixed(2);
    refs.calcS.textContent = "0.00";
    refs.calcPred.textContent = "+1";
    refs.calcScoreEq.textContent = "s = w1*x1 + w2*x2 + b";
    refs.calcUpdateEq.textContent = "w <- w + eta*y*x, b <- b + eta*y";
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
  refs.gridSizeEl.addEventListener("change", () => {
    state.gridSize = Number(refs.gridSizeEl.value);
    state.gridInput = makeInputGrid(state.gridSize);
    render(refs, state);
  });
  window.addEventListener("keydown", (event) => {
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

  updateLrDisplay();
  render(refs, state);
  void reset();
}
