export type Point = { x: number; y: number; label: 1 | -1 };
export type Grid = number[][];

export type LastStep = {
  x: number[];
  y: number;
  score: number;
  pred: number;
  mistake: boolean;
  deltaW: number[];
  deltaB: number;
  lr: number;
};

export type CustomSample = { grid: Grid; y: number };

export type CustomConfig = {
  rows: number;
  cols: number;
  samples: CustomSample[];
};

export type TooltipState = {
  visible: boolean;
  text: string;
  x: number;
  y: number;
};
