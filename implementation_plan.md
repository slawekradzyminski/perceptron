# Implementation Plan — Error + Multi‑Layer Network

Last updated: 2026-01-01

## Phases

### Phase 1 — Error/Loss Foundations (done)
Goal: make error calculations explicit and testable.
- Add activation functions + derivatives.
- Add loss functions (perceptron, MSE, BCE) + gradients.
- Unit tests for all math primitives.

### Phase 2 — Logistic Neuron + Adaline (done)
Goal: replace hard-step with differentiable units.
- Logistic neuron model (sigmoid + BCE).
- Adaline model (linear + MSE).
- Training runner for loss curves.
- Unit tests for models + runner.

### Phase 3 — 2‑Layer MLP (XOR) (done)
Goal: demonstrate multi-layer learning.
- Dense layers + backprop.
- Hidden activations heatmaps (planned).
- Decision boundary visualization (planned).

### Phase 4 — Grid‑shape MLP (done)
Goal: apply multi-layer learning to grid inputs.
- Flattened grid input.
- Visualize weight templates per neuron.
- Gradient maps for learning intuition (planned).

## Progress
- [x] Phase 1 — Error/Loss Foundations
- [x] Phase 2 — Logistic Neuron + Adaline
- [x] Phase 3 — 2‑Layer MLP (XOR)
- [x] Phase 4 — Grid‑shape MLP

## Notes
- Keep code deterministic and unit-tested (pytest).
- Frontend remains visualization-only; backend is the single source of truth.
