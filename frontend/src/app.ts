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
    !apiBase
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
}

async function apiStep(refs: AppRefs, state: State) {
  const res = await fetch(`${refs.apiBase.value}/step`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset: state.datasetName }),
  });
  if (!res.ok) return;
  const data = await res.json();
  state.w = data.w;
  state.b = data.b;
  state.sampleIdx = data.idx ?? state.sampleIdx;
}

async function apiReset(refs: AppRefs, state: State) {
  const res = await fetch(`${refs.apiBase.value}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset: state.datasetName }),
  });
  if (!res.ok) return;
  const data = await res.json();
  state.w = data.w;
  state.b = data.b;
  state.sampleIdx = data.idx ?? 0;
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
    render(refs, state);
  });
  refs.gridSizeEl.addEventListener("change", () => {
    state.gridSize = Number(refs.gridSizeEl.value);
    state.gridInput = makeInputGrid(state.gridSize);
    render(refs, state);
  });

  updateLrDisplay();
  render(refs, state);
}
