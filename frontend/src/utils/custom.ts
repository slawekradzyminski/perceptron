import type { CustomConfig, Point } from "../types";
import { createFilledGrid, flattenGrid } from "./visuals";

export function defaultCustomConfig(): CustomConfig {
  return {
    rows: 2,
    cols: 2,
    samples: [{ grid: createFilledGrid(2, 2, -1), y: -1 }],
  };
}

export function pointsForDataset(
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

export function buildCustomPayload(customConfig: CustomConfig) {
  return {
    grid_rows: customConfig.rows,
    grid_cols: customConfig.cols,
    samples: customConfig.samples.map((sample) => ({
      grid: sample.grid,
      y: sample.y,
    })),
  };
}
