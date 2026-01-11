/**
 * Centralized API Types Module
 *
 * This module provides a single source of truth for all API types,
 * using prefixed interfaces to match backend API structure.
 */

// ==============================================================================
// Deep Learning API Types (Chapter 4)
// ==============================================================================

export interface DeepArchitecture {
  input_dim: number;
  hidden_dims: number[];
  output_dim: number;
  depth: number;
  width: number;
  param_count: number;
}

export interface DeepMetrics {
  loss: number;
  accuracy: number;
}

export interface DeepDatasetInfo {
  name: string;
  description: string;
  n_classes: number;
}

export interface DeepSample {
  x: number[];
  y: number;
}

export interface DeepState {
  dataset: string;
  dataset_info: DeepDatasetInfo;
  sample_count: number;
  idx: number;
  epoch: number;
  total_steps: number;
  lr: number;
  architecture: DeepArchitecture;
  metrics: DeepMetrics;
  samples: DeepSample[];
}

export interface DeepStepInfo {
  batch_size: number;
  last_loss: number;
  last_correct: boolean;
  last_prediction: number;
  grad_norm: number;
}

export interface DeepStepResponse extends DeepState {
  step_info: DeepStepInfo;
}

export interface DeepBoundary {
  resolution: number;
  predictions: number[][];
  region_ids: number[][];
  region_count: number;
  theoretical_max: number;
}

export interface DeepRegions {
  count: number;
  theoretical_max: number;
  efficiency: number;
}

export interface DeepHistoryEntry {
  step: number;
  epoch: number;
  loss: number;
  accuracy: number;
}

export interface DeepComparisonEntry {
  depth: number;
  width: number;
  hidden_dims: number[];
  param_count: number;
  actual_regions: number;
  theoretical_max: number;
  accuracy: number;
  loss: number;
  total_steps: number;
}

export interface DeepResetOptions {
  dataset?: string;
  hidden_dims?: number[];
  lr?: number;
  seed?: number;
}

// ==============================================================================
// Glass Box API Types (Chapters 7 & 8)
// ==============================================================================

export interface AttentionResult {
  text: string;
  tokens: string[];
  n_layers: number;
  n_heads: number;
  seq_len: number;
  model_name: string;
  attentions: number[][][][]; // [layers][heads][seq][seq]
  explanation: string;
}

export interface LogitLensPrediction {
  token: string;
  token_id: number;
  probability: number;
}

export interface LogitLensLayer {
  layer: number;
  layer_name: string;
  predictions: LogitLensPrediction[];
}

export interface LogitLensResult {
  text: string;
  tokens: string[];
  n_layers: number;
  top_k: number;
  model_name: string;
  layers: LogitLensLayer[];
  explanation: string;
}

export interface KVCacheArchitecture {
  name: string;
  full_name: string;
  description: string;
  memory_bytes: number;
  memory_formatted: string;
  ratio_to_mha: number;
  kv_heads?: number;
  latent_dim?: number;
}

export interface KVCacheConfig {
  context_length: number;
  n_layers: number;
  d_model: number;
  n_heads: number;
  d_head: number;
  gqa_groups: number;
  mla_latent_dim: number;
  bytes_per_param: number;
}

export interface KVCacheResult {
  config: KVCacheConfig;
  architectures: KVCacheArchitecture[];
  mha_to_mla_savings: number;
  explanation: string;
}

export interface KVCacheRequestConfig {
  context_length?: number;
  n_layers?: number;
  d_model?: number;
  n_heads?: number;
  gqa_groups?: number;
  mla_latent_dim?: number;
}

// ==============================================================================
// Perceptron API Types
// ==============================================================================

export interface PerceptronState {
  w: number[];
  b: number;
  idx: number;
  dataset: string;
  lr: number;
  next_x: number[];
  next_y: number;
  grid_rows: number;
  grid_cols: number;
  sample_count: number;
}

export interface PerceptronStepResponse extends PerceptronState {
  x: number[];
  y: number;
  score: number;
  pred: number;
  mistake: boolean;
  delta_w: number[];
  delta_b: number;
}

// ==============================================================================
// Common API Types
// ==============================================================================

export interface ApiError {
  detail: string;
}

export interface ClearedResponse {
  cleared: boolean;
}
