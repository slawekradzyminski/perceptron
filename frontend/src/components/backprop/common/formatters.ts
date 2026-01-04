export function formatNum(value: number, digits = 4) {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

export function formatVec(values: number[], digits = 4) {
  return `[${values.map((val) => formatNum(val, digits)).join(", ")}]`;
}

export function formatMatrix(matrix: number[][], digits = 4) {
  return `[${matrix.map((row) => formatVec(row, digits)).join(", ")}]`;
}
