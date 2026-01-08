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

## Deep Learning endpoints
- `GET /deep/state`
  - Returns current deep MLP state (architecture, metrics, samples).
- `POST /deep/reset`
  - Configure architecture and dataset. Body: `{hidden_dims, dataset, lr, seed}`.
- `POST /deep/step`
  - Train for one or more steps. Body: `{batch_size}`.
- `POST /deep/epoch`
  - Train for one full epoch.
- `GET /deep/regions?resolution=50`
  - Returns region count, theoretical max, and efficiency.
- `GET /deep/boundary?resolution=50`
  - Returns decision boundary and region tiling grids.
- `POST /deep/comparison/add`
  - Add current architecture to comparison table.
- `GET /deep/comparison`
  - Returns comparison table entries.
- `POST /deep/comparison/clear`
  - Clears the comparison table.

## AlexNet endpoints (Chapter 5)
- `GET /alexnet/state`
  - Returns current layer, sample images, layer info, and param count.
- `GET /alexnet/layers`
  - Returns information about all 5 convolutional layers.
- `POST /alexnet/layer/{layer}`
  - Set the current layer for exploration (1-5).
- `GET /alexnet/filters?layer=1&max_filters=64`
  - Returns filter visualizations as base64-encoded images.
- `GET /alexnet/activations?sample=gradient&layer=1&max_activations=64`
  - Returns activation maps for a sample image.
- `POST /alexnet/activations?layer=1&max_activations=64`
  - Upload an image and get activation maps (multipart/form-data).
- `GET /alexnet/sample/{name}`
  - Returns a sample test image as base64 (gradient, checkerboard, noise).

## Transformer endpoints (Chapter 5)
- `GET /transformer/state`
  - Returns current text, token count, encoding info.
- `POST /transformer/tokenize`
  - Tokenize text. Body: `{text}`.
- `POST /transformer/embed`
  - Get embedding dimension info. Body: `{text}` (optional).
- `POST /transformer/trace`
  - Show matrix flow through transformer blocks. Body: `{text, n_blocks}`.
- `POST /transformer/generate`
  - Simulate autoregressive generation. Body: `{prompt, max_steps}`.
- `GET /transformer/models`
  - Returns GPT model comparison data.
- `GET /transformer/scale`
  - Returns all models for scale comparison (LeNet-5 to GPT-4).
- `GET /transformer/scale/cnns`
  - Returns only CNN models.
- `GET /transformer/scale/transformers`
  - Returns only Transformer models.
- `GET /transformer/scale/compare?model1=AlexNet&model2=GPT-4`
  - Compare two models by name.
- `GET /transformer/scale/growth`
  - Returns exponential growth data for visualization.
