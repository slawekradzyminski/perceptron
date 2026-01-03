type ErrorSurfaceOverrides = Partial<{
  dataset: string;
  grid_rows: number;
  grid_cols: number;
  steps: number;
  w_range: [number, number];
  bias: number;
  sample_count: number;
  grid: number[][];
}>;

type MlpInternalsOverrides = Partial<{
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
}>;

export const mockPerceptronState = (overrides: Partial<{
  w: number[];
  b: number;
  idx: number;
  dataset: string;
  next_x: number[];
  next_y: number;
  grid_rows: number;
  grid_cols: number;
  sample_count: number;
}> = {}) => ({
  w: [0, 0],
  b: 0,
  idx: 0,
  dataset: "or",
  next_x: [-1, -1],
  next_y: -1,
  grid_rows: 1,
  grid_cols: 2,
  sample_count: 4,
  ...overrides,
});

export const mockPerceptronStep = () => ({
  w: [1, 0],
  b: 1,
  idx: 1,
  dataset: "or",
  x: [1, -1],
  y: 1,
  score: 1,
  pred: 1,
  mistake: false,
  delta_w: [0, 0],
  delta_b: 0,
  lr: 1,
  next_x: [-1, 1],
  next_y: 1,
  grid_rows: 1,
  grid_cols: 2,
  sample_count: 4,
});

export const mockErrorSurface = (overrides: ErrorSurfaceOverrides = {}) => ({
  dataset: "or",
  grid_rows: 1,
  grid_cols: 2,
  steps: 3,
  w_range: [-1, 1],
  bias: 0,
  sample_count: 4,
  grid: [
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5],
  ],
  ...overrides,
});

export const mockMlpInternals = (overrides: MlpInternalsOverrides = {}) => ({
  dataset: "or",
  grid_rows: 1,
  grid_cols: 2,
  hidden_dim: 2,
  sample_index: 0,
  sample_count: 4,
  x: [1, -1],
  y: 1,
  y01: 1,
  loss: 0.2,
  p_hat: 0.6,
  hidden: {
    weights_before: [[0.1, 0.2], [0.3, 0.4]],
    bias_before: [0, 0],
    weights_after: [[0.2, 0.3], [0.4, 0.5]],
    bias_after: [0.1, 0.1],
    z: [0.1, 0.2],
    a: [0.1, 0.2],
    templates_before: [[[0.1, 0.2]], [[0.3, 0.4]]],
    templates_after: [[[0.2, 0.3]], [[0.4, 0.5]]],
  },
  output: {
    weights_before: [[0.2, 0.3]],
    bias_before: [0],
    weights_after: [[0.3, 0.4]],
    bias_after: [0.1],
    z: 0.2,
    a: 0.6,
  },
  gradients: {
    hidden_W: [[0.1, 0.1], [0.2, 0.2]],
    hidden_b: [0.1, 0.1],
    out_W: [[0.1, 0.1]],
    out_b: [0.1],
    templates: [[[0.1, 0.1]], [[0.2, 0.2]]],
  },
  ...overrides,
});

export const mockLmsReset = () => ({
  w: [0, 0],
  b: 0,
  idx: 0,
  lr: 0.1,
  x: [-1, -1],
  y: -1,
  sample_count: 4,
  dataset: "or",
});

export const mockMlpReset = () => ({
  dataset: "xor",
  grid_rows: 1,
  grid_cols: 2,
  hidden_dim: 2,
  lr: 0.5,
  seed: 0,
  idx: 0,
  sample_count: 4,
  next_x: [-1, -1],
  next_y: -1,
  hidden: { weights: [[0, 0], [0, 0]], bias: [0, 0], templates: [[[0, 0]], [[0, 0]]] },
  output: { weights: [[0, 0]], bias: [0] },
  evals: [],
});

export const mockMlpSnapshot = (overrides: Partial<ReturnType<typeof baseMlpSnapshot>> = {}) => {
  const base = baseMlpSnapshot();
  return { ...base, ...overrides };
};

const baseMlpSnapshot = () => ({
  dataset: "xor",
  grid_rows: 1,
  grid_cols: 2,
  hidden_dim: 2,
  lr: 0.5,
  seed: 0,
  idx: 0,
  sample_count: 4,
  next_x: [-1, -1],
  next_y: -1,
  hidden: {
    weights: [[0.32, 0.28], [-0.1, -0.22]],
    bias: [0.02, 0.02],
    templates: [[[0.32, 0.28]], [[-0.1, -0.22]]],
  },
  output: { weights: [[0.34, 0.25]], bias: [0.06] },
  evals: [
    { x: [-1, 1], y: 1, p_hat: 0.48, pred: -1 },
    { x: [1, -1], y: 1, p_hat: 0.52, pred: 1 },
  ],
});
