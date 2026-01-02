export function getCellFromEvent(
  event: React.MouseEvent<HTMLCanvasElement>,
  rows: number,
  cols: number,
) {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const col = Math.min(cols - 1, Math.max(0, Math.floor((x / rect.width) * cols)));
  const row = Math.min(rows - 1, Math.max(0, Math.floor((y / rect.height) * rows)));
  return { row, col };
}
