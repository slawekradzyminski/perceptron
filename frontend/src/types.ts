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

export type LmsState = {
  w: number[];
  b: number;
  idx: number;
  lr: number;
  x: number[];
  y: number;
  sample_count: number;
  dataset: string;
};

export type LmsStep = {
  x: number[];
  y: number;
  w_before: number[];
  b_before: number;
  y_hat: number;
  error: number;
  grad_w1: number;
  grad_w2: number;
  grad_b: number;
  w_after: number[];
  b_after: number;
  idx: number;
  lr: number;
};

export type ErrorSurfaceResponse = {
  dataset: string;
  grid_rows: number;
  grid_cols: number;
  steps: number;
  w_range: [number, number];
  bias: number;
  sample_count: number;
  grid: number[][];
};

export type MlpInternalsResponse = {
  dataset: string;
  grid_rows: number;
  grid_cols: number;
  hidden_dim: number;
  sample_index: number;
  sample_count: number;
  x: number[];
  y: number;
  y01: number;
  loss: number;
  p_hat: number;
  hidden: {
    weights_before: number[][];
    bias_before: number[];
    weights_after: number[][];
    bias_after: number[];
    z: number[];
    a: number[];
    templates_before: number[][][];
    templates_after: number[][][];
  };
  output: {
    weights_before: number[][];
    bias_before: number[];
    weights_after: number[][];
    bias_after: number[];
    z: number;
    a: number;
  };
  gradients: {
    hidden_W: number[][];
    hidden_b: number[];
    out_W: number[][];
    out_b: number[];
    templates: number[][][];
  };
};
