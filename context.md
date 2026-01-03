# Project Context

## Sources
- DEV article: "The Perceptron — Visually See an AI Learn" (interactive grid/weights demo)
- Welch Labs exercises PDF (perceptron learning rule, geometry)
- Welch Labs repo (chapter notebooks, reference structure)

Links
- https://dev.to/grahamthedev/the-perceptron-visually-see-an-ai-learn-2fd9
- https://www.welchlabs.com/s/exercises_rev_12_nov_14_1.pdf
- https://github.com/stephencwelch/ai_book

## Goal
Build a codebase that helps me understand how a perceptron works and how the math behaves under the hood, via visualization and step-by-step updates.

## Core Algorithm (Rosenblatt Perceptron)
- Inputs: vector x in R^d
- Parameters: weights w in R^d, bias b in R
- Score: s = w·x + b
- Prediction: y_hat = sign(s), with labels y in {-1, +1}

Perceptron learning rule (mistake-driven):
- If y * (w·x + b) <= 0:
  - w <- w + lr * y * x
  - b <- b + lr * y

## Visual Targets
- 2D scatter (OR vs XOR) with a moving decision boundary
- Grid-based input/weight/contribution panels ("switchboard" metaphor)
- A score meter showing the running sum
- Optional: translate shapes across the grid to force generalization

## Expected Milestones
1. 2D perceptron with live boundary animation
2. Grid/weight-board visualization (input, weights, contributions, score meter)
3. Training on translated patterns + mistakes/accuracy tracking
4. Optional: Adaline/LMS comparison for gradient descent intuition

## UI Controls
- Dataset selector: OR / XOR / grid shapes
- Learning rate slider
- Step / Play / Reset
- Toggles: show contributions, train on translations

## Deliverables
- Python core (perceptron, datasets, metrics)
- TypeScript frontend for interactive visualization (if needed)

## Backend API
- Entrypoint: `perceptron-api` (`backend/api_server.py`)
- Core endpoints:
  - `GET /state` — current perceptron state.
  - `POST /step` — advance one training step.
  - `POST /reset` — reset state and/or swap dataset.
  - `POST /error-surface` — MSE surface for 2D inputs (grid_rows * grid_cols == 2).
  - `POST /mlp-internals` — 2-layer MLP internals (weights, activations, gradients).

### Diagnostics Endpoints
`POST /error-surface`
- Request fields: `dataset` (or/xor/custom), `steps`, `w_min`, `w_max`, `b`.
- Custom datasets require `grid_rows`, `grid_cols`, `samples`.
- Response: `{ dataset, grid_rows, grid_cols, steps, w_range, bias, sample_count, grid }`.

`POST /mlp-internals`
- Request fields: `dataset`, `hidden_dim`, `sample_index`, `lr`, `seed`.
- Custom datasets require `grid_rows`, `grid_cols`, `samples`.
- Response: `{ dataset, grid_rows, grid_cols, hidden_dim, sample_index, sample_count, x, y, y01, loss, p_hat, hidden, output, gradients }`.
