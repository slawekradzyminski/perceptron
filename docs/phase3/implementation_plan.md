# Implementation Plan (Chapter 3) — Complete

This chapter’s lab is now fully integrated into the existing Perceptron Visual Lab (backend + frontend). The plan below documents what is implemented and where it lives.

## Core UX (routes + pages)
- `/backprop/tinygps`
  - TinyGPS controls + exercise setup
  - Book Table with per-cell hover math
  - Loss Trend chart
  - Backprop Flow diagram with live values
  - City Map + decision boundary for Paris/Berlin (1D longitude model)
- `/backprop/regression`
  - MSE + L1 regression controls
  - Dataset editor
  - Book presets (MSE/L1)
  - Regression plot (step 0/1/7 overlays)
  - Regression history + Book Table Mode (book-literal option for L1 discrepancy)

## Backend: APIs and services
### Routes
- `GET /backprop/state`
- `POST /backprop/tinygps/reset`
- `POST /backprop/tinygps/step`
- `POST /backprop/regression/reset`
- `POST /backprop/regression/step`

### Service + math
- `backend/services/backprop_service.py`
  - TinyGPS state, datasets, params, sample ordering, metrics
  - Regression state (MSE/L1), params, sample ordering
- `backend/nn/tinygps.py`
  - Softmax + cross-entropy forward/backward (1D + 2D)
- `backend/nn/regression.py`
  - Single neuron regression (MSE + L1) forward/backward
- `backend/core/datasets.py`
  - City coordinate fixtures + exercise subsets

## Frontend: components + hooks
- TinyGPS:
  - Controls, step/setup, loss trend, backprop diagram, city map, book table
- Regression:
  - Controls, dataset editor, plot, book table, history, presets
- Hooks:
  - `useBackpropApi` for state + step/reset
  - `useHotkeys` for S/R

## Exercise coverage
- TinyGPS: exercises 3.4–3.17 (forward pass, gradients, GD steps, loss/accuracy tables)
- Regression: MSE + L1 exercises (step tables, line plots, convergence intuition)
- Conceptual Q&A: exercises 3.1–3.3 in `docs/phase3/qa.md`

## Optional extensions
- Optional Map of Language demo (see `docs/phase3/map_of_language.md`)
