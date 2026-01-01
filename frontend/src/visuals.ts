export type Grid = number[][];
export type Point = { x: number; y: number; label: 1 | -1 };

export function valueColor(v: number): string {
  if (v === 0) return "#ffffff";
  return v > 0 ? "#c7f9cc" : "#f6b6b0";
}

export function makeGrid(rows: number, cols: number, fill: (r: number, c: number) => number): Grid {
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => fill(r, c)),
  );
}

export function createFilledGrid(rows: number, cols: number, value: number): Grid {
  return makeGrid(rows, cols, () => value);
}

export function resizeGrid(grid: Grid, rows: number, cols: number, fill: number): Grid {
  return makeGrid(rows, cols, (r, c) => grid[r]?.[c] ?? fill);
}

export function flattenGrid(grid: Grid): number[] {
  return grid.flat();
}

export function reshapeToGrid(values: number[], rows: number, cols: number): Grid {
  return makeGrid(rows, cols, (r, c) => values[r * cols + c] ?? 0);
}

export function contributionGrid(a: Grid, b: Grid): Grid {
  return a.map((row, r) => row.map((v, c) => v * b[r][c]));
}

export function setGridCanvasSize(canvas: HTMLCanvasElement, rows: number, cols: number) {
  const cellSize = 60;
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  canvas.style.aspectRatio = `${cols} / ${rows}`;
}

export function drawGrid(canvas: HTMLCanvasElement, grid: Grid, palette: (v: number) => string) {
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

export function drawBoundary(ctx: CanvasRenderingContext2D, w: number[], b: number) {
  if (w.length < 2) {
    return;
  }
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
  if (Math.abs(w[1]) < 1e-6) {
    const xLine = -b / (w[0] === 0 ? 1e-6 : w[0]);
    const mapX = (x: number) => ((x + 1.5) / 3) * width;
    ctx.strokeStyle = "#1c1b19";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(mapX(xLine), 0);
    ctx.lineTo(mapX(xLine), height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = "#c9b79f";
    ctx.lineWidth = 2;
    ctx.strokeRect(4, 4, width - 8, height - 8);
    return;
  }
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

export function drawPoints(ctx: CanvasRenderingContext2D, points: Point[]) {
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

export function formatVec(v: number[]) {
  return `[${v.map((val) => val.toFixed(2)).join(", ")}]`;
}

export function formatSign(val: number) {
  return val >= 0 ? `+${val.toFixed(2)}` : `${val.toFixed(2)}`;
}
