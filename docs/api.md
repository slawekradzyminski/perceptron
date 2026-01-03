# Backend API (FastAPI)

Base URL: `http://127.0.0.1:8000`

## Core perceptron endpoints
- `GET /state`
  - Returns current weights, bias, dataset, next sample, grid size.
- `POST /step`
  - Advances one training step and returns the step diagnostics.
  - Optional body fields: `dataset`, `lr`, `grid_rows`, `grid_cols`, `samples` (for custom).
- `POST /reset`
  - Resets weights and sample index (and optionally dataset).

### Dataset formats
- `dataset`: `"or"` | `"xor"` | `"custom"`
- `grid_rows`, `grid_cols`: integers in [1, 5]
- `samples`: list of `{ x: number[], y: -1|1 }` or `{ grid: number[][], y: -1|1 }`
  - When `grid` is used, `x` is derived by row-major flattening.

## Diagnostics endpoints
- `POST /error-surface`
  - Computes MSE across a (w1, w2) grid.
  - Requires 2D inputs (`grid_rows * grid_cols == 2`).
  - Body fields: `dataset`, `steps`, `w_min`, `w_max`, `b`, `grid_rows`, `grid_cols`, `samples`.
- `POST /mlp-internals`
  - Returns hidden/output weights, activations, and gradients for a 1-hidden-layer MLP.
  - Body fields: `dataset`, `hidden_dim`, `sample_index`, `lr`, `seed`, `grid_rows`, `grid_cols`, `samples`.

## LMS endpoints
- `GET /lms/state`
  - Returns current LMS weights, bias, sample index.
- `POST /lms/step`
  - Advances one LMS step and returns gradients and updated parameters.
- `POST /lms/reset`
  - Resets LMS weights and sample index.
