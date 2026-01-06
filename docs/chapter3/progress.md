# Phase 3 Progress

## Status (2026-01-06)
- Chapter 3 backprop lab fully implemented in `/backprop/tinygps` and `/backprop/regression`.
- TinyGPS:
  - Exercise datasets restored: `paris-berlin-4`, `paris-madrid-4` (2 samples per city).
  - Full datasets preserved: `paris-berlin`, `paris-madrid`, `madrid-paris-berlin`, `four-cities`.
  - Book Table view with per-cell hover explanations (h, softmax, gradients, loss).
  - Loss Trend chart (loss over steps) with labeled axes.
  - Backprop Flow diagram (live values + gradient flow animation + hover calculus).
  - City Map + boundary for Paris/Berlin (1D longitude boundary rendered as a vertical line).
  - Hotkeys: **S** to step, **R** to reset.
- Regression:
  - MSE + L1 modes, dataset editor, book presets, and full sample list in API state.
  - Regression plot with step 0/1/7 overlays + scatter.
  - Book Table Mode with display rules + optional “book-literal” loss (L1 step 1 discrepancy).
- Backend:
  - Backprop service + `/backprop` API routes for TinyGPS + regression.
  - Tests for TinyGPS/regression math, datasets, and API routes.
- Map of Language: preserved in repo with docs at `docs/chapter3/map_of_language.md`.
