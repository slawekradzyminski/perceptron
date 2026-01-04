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

export type MlpTrainerEval = {
  x: number[];
  y: number;
  p_hat: number;
  pred: number;
};

export type MlpTrainerSnapshot = {
  dataset: string;
  grid_rows: number;
  grid_cols: number;
  hidden_dim: number;
  lr: number;
  seed: number;
  idx: number;
  sample_count: number;
  next_x: number[];
  next_y: number;
  hidden: {
    weights: number[][];
    bias: number[];
    templates: number[][][];
  };
  output: {
    weights: number[][];
    bias: number[];
  };
  evals: MlpTrainerEval[];
};

export type MlpTrainerResponse = MlpTrainerSnapshot & {
  step?: MlpInternalsResponse;
};

export type GdLossPoint = {
  p: number;
  l1: number;
  ce: number;
};

export type GdLossCurvesResponse = {
  points: GdLossPoint[];
};

export type TokenLossRow = {
  context: string;
  correct_token: string;
  p_correct: number;
  top_token: string;
  top_prob: number;
  l1_loss: number;
  ce_loss: number;
};

export type TokenLossExample = {
  id: string;
  title: string;
  description: string;
  average: { l1: number; ce: number };
  rows: TokenLossRow[];
};

export type TokenLossResponse = {
  examples: TokenLossExample[];
  source?: string;
  warning?: string;
};

export type OllamaStatusResponse = {
  ok: boolean;
  model?: string;
  base_url?: string;
  version?: string;
  models?: string[];
  last_latency_ms?: number | null;
  error?: string;
};

export type PromptLogitsToken = {
  token: string;
  prob: number;
  logprob: number;
  rank: number;
};

export type PromptLogitsResponse = {
  prompt: string;
  tokens: PromptLogitsToken[];
  warning?: string;
};

export type TinyGpsSample = {
  x: number[];
  y: number;
  city: string;
  lat: number;
  lon: number;
};

export type TinyGpsParams = {
  m: number[];
  b: number[];
  M: number[][];
};

export type TinyGpsState = {
  dataset: string;
  mode: "1d" | "2d";
  cities: string[];
  idx: number;
  lr: number;
  sample_count: number;
  feature_order: string[];
  sample_order: number[];
  params: TinyGpsParams;
  next_sample: TinyGpsSample;
  samples: TinyGpsSample[];
};

export type RegressionState = {
  loss: "mse" | "l1";
  idx: number;
  lr: number;
  sample_count: number;
  params: { m: number; b: number };
  next_sample: { x: number; y: number };
  samples: { x: number; y: number }[];
  sample_order: number[];
};

export type BackpropState = {
  tinygps: TinyGpsState;
  regression: RegressionState;
  tinygps_datasets: string[];
  tinygps_city_coords: Record<string, [number, number][]>;
};

export type TinyGpsStep = {
  dataset: string;
  mode: "1d" | "2d";
  cities: string[];
  idx: number;
  sample_idx: number;
  lr: number;
  feature_order: string[];
  sample: TinyGpsSample;
  logits: number[];
  probs: number[];
  loss: number;
  pred: number;
  correct: boolean;
  grads: { m?: number[]; M?: number[][]; b: number[] };
  params_before: TinyGpsParams;
  params_after: TinyGpsParams;
  metrics_after: { loss: number; accuracy: number };
  sample_count: number;
};

export type RegressionStep = {
  loss: "mse" | "l1";
  idx: number;
  sample_idx: number;
  lr: number;
  sample: { x: number; y: number };
  y_hat: number;
  loss_value: number;
  grads: { m: number; b: number };
  params_before: { m: number; b: number };
  params_after: { m: number; b: number };
  mean_loss: number;
  sample_count: number;
};
