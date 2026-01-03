import type { Grid, Point } from "../types";

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

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function interpolateColor(start: string, end: string, t: number) {
  const s = start.replace("#", "");
  const e = end.replace("#", "");
  const sr = parseInt(s.slice(0, 2), 16);
  const sg = parseInt(s.slice(2, 4), 16);
  const sb = parseInt(s.slice(4, 6), 16);
  const er = parseInt(e.slice(0, 2), 16);
  const eg = parseInt(e.slice(2, 4), 16);
  const eb = parseInt(e.slice(4, 6), 16);
  const r = Math.round(sr + (er - sr) * t);
  const g = Math.round(sg + (eg - sg) * t);
  const b = Math.round(sb + (eb - sb) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export function heatColor(value: number, minValue: number, maxValue: number) {
  if (maxValue <= minValue) return "#fff6e8";
  const t = clamp((value - minValue) / (maxValue - minValue), 0, 1);
  return interpolateColor("#fff6e8", "#c9574c", t);
}

export function setHeatmapCanvasSize(canvas: HTMLCanvasElement, rows: number, cols: number) {
  const cellSize = 12;
  canvas.width = cols * cellSize;
  canvas.height = rows * cellSize;
  canvas.style.aspectRatio = `${cols} / ${rows}`;
}

export function drawHeatmap(
  canvas: HTMLCanvasElement,
  grid: number[][],
  minValue: number,
  maxValue: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const rows = grid.length;
  const cols = grid[0].length;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = heatColor(grid[r][c], minValue, maxValue);
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }
  ctx.strokeStyle = "#e2cfba";
  ctx.lineWidth = 1;
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

export function drawMlpBoundary(
  ctx: CanvasRenderingContext2D,
  points: Point[],
  hiddenW: number[][],
  hiddenB: number[],
  outW: number[],
  outB: number,
) {
  if (hiddenW.length === 0 || hiddenW[0]?.length < 2) {
    return;
  }
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  const xMin = -1.5;
  const xMax = 1.5;
  const yMin = -1.5;
  const yMax = 1.5;
  const steps = 48;
  const dx = (xMax - xMin) / steps;
  const dy = (yMax - yMin) / steps;
  const mapX = (x: number) => ((x - xMin) / (xMax - xMin)) * width;
  const mapY = (y: number) => height - ((y - yMin) / (yMax - yMin)) * height;

  const predict = (x1: number, x2: number) => {
    const hidden = hiddenW.map((w, i) => Math.tanh(w[0] * x1 + w[1] * x2 + (hiddenB[i] ?? 0)));
    const z = hidden.reduce((sum, a, i) => sum + (outW[i] ?? 0) * a, outB);
    return 1 / (1 + Math.exp(-z));
  };

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

  const field: number[][] = [];
  for (let j = 0; j <= steps; j++) {
    const row: number[] = [];
    const y = yMin + j * dy;
    for (let i = 0; i <= steps; i++) {
      const x = xMin + i * dx;
      row.push(predict(x, y) - 0.5);
    }
    field.push(row);
  }

  const cases: number[][][] = [
    [],
    [[3, 2]],
    [[2, 1]],
    [[3, 1]],
    [[0, 1]],
    [[0, 3], [2, 1]],
    [[0, 2]],
    [[0, 3]],
    [[0, 3]],
    [[0, 2]],
    [[0, 1], [2, 3]],
    [[0, 1]],
    [[1, 3]],
    [[2, 1]],
    [[3, 2]],
    [],
  ];

  const edgePoint = (
    edge: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    tl: number,
    tr: number,
    br: number,
    bl: number,
  ) => {
    const interp = (a: number, b: number, va: number, vb: number) => {
      if (va === vb) return (a + b) / 2;
      return a + (b - a) * (va / (va - vb));
    };
    switch (edge) {
      case 0: {
        const x = interp(x0, x1, tl, tr);
        return { x, y: y1 };
      }
      case 1: {
        const y = interp(y1, y0, tr, br);
        return { x: x1, y };
      }
      case 2: {
        const x = interp(x1, x0, br, bl);
        return { x, y: y0 };
      }
      case 3: {
        const y = interp(y0, y1, bl, tl);
        return { x: x0, y };
      }
      default:
        return { x: x0, y: y0 };
    }
  };

  ctx.strokeStyle = "#1c1b19";
  ctx.lineWidth = 2.5;
  for (let j = 0; j < steps; j++) {
    const y0 = yMin + j * dy;
    const y1 = yMin + (j + 1) * dy;
    for (let i = 0; i < steps; i++) {
      const x0 = xMin + i * dx;
      const x1 = xMin + (i + 1) * dx;
      const bl = field[j][i];
      const br = field[j][i + 1];
      const tr = field[j + 1][i + 1];
      const tl = field[j + 1][i];
      const idx =
        (tl >= 0 ? 8 : 0) +
        (tr >= 0 ? 4 : 0) +
        (br >= 0 ? 2 : 0) +
        (bl >= 0 ? 1 : 0);
      const segments = cases[idx];
      if (!segments || segments.length === 0) continue;
      segments.forEach(([e1, e2]) => {
        const p1 = edgePoint(e1, x0, y0, x1, y1, tl, tr, br, bl);
        const p2 = edgePoint(e2, x0, y0, x1, y1, tl, tr, br, bl);
        ctx.beginPath();
        ctx.moveTo(mapX(p1.x), mapY(p1.y));
        ctx.lineTo(mapX(p2.x), mapY(p2.y));
        ctx.stroke();
      });
    }
  }

  const drawHiddenBoundary = (w: number[], b: number, color: string) => {
    if (w.length < 2) return;
    const x1 = -1.5;
    const x2 = 1.5;
    if (Math.abs(w[1]) < 1e-6) {
      const xLine = -b / (w[0] === 0 ? 1e-6 : w[0]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(mapX(xLine), 0);
      ctx.lineTo(mapX(xLine), height);
      ctx.stroke();
      ctx.setLineDash([]);
      return;
    }
    const y1 = -(w[0] * x1 + b) / w[1];
    const y2 = -(w[0] * x2 + b) / w[1];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.moveTo(mapX(x1), mapY(y1));
    ctx.lineTo(mapX(x2), mapY(y2));
    ctx.stroke();
    ctx.setLineDash([]);
  };

  const hiddenColors = ["#2a7b8f", "#2f7d4f", "#8a5cc5", "#c46a37", "#2c4b9b", "#a2476b"];
  hiddenW.forEach((w, idx) => {
    drawHiddenBoundary(w, hiddenB[idx] ?? 0, hiddenColors[idx % hiddenColors.length]);
  });

  ctx.strokeStyle = "#c9b79f";
  ctx.lineWidth = 2;
  ctx.strokeRect(4, 4, width - 8, height - 8);
  drawPoints(ctx, points);
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
