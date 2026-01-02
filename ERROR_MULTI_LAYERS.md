# Error + Multi-Layer Network Plan

This document outlines how we extend the current perceptron demo to include
explicit error calculations and multi-layer networks (MLP). The goal is to keep
the “lab bench” feel while making the math visible and testable.

## 0) Key Ideas

### A) Error / Loss (what you minimize)

We will compute and visualize loss functions alongside updates.

- **Perceptron loss (hinge-like):**
  - loss = max(0, -y * (w·x + b))
  - zero when correct with margin, linear penalty when wrong.

- **MSE (Adaline):**
  - y_hat = w·x + b
  - loss = 0.5 * (y - y_hat)^2
  - smooth, gradient-friendly.

- **Binary cross-entropy (logistic neuron):**
  - p_hat = sigmoid(z), z = w·x + b
  - loss = -(y * log(p_hat) + (1 - y) * log(1 - p_hat))
  - standard for binary classification.

### B) Multi-perceptron network (MLP)

A multi-layer network requires differentiable activations and backprop.

- **Hidden layers:** tanh or ReLU
- **Output:** sigmoid (binary)
- **Training:** forward → loss → backward (gradients) → update

## 1) Suggested Structure

```
backend/nn/
  activations.py
  losses.py
  layers.py
  optim.py
  models.py
backend/viz/
  viz_error_surface.py
  viz_mlp_internals.py
```

## 2) Milestones

### Milestone A — Baseline perceptron (done)
- Mistake-driven updates
- Decision boundary animation

### Milestone B — Error calculations (Phase 1)
- Add perceptron loss, MSE, BCE
- Plot loss curves per step
- Add error surface for MSE (2D case)

### Milestone C — Two-layer MLP for XOR
- Input(2) → Hidden(2–4) → Output(1)
- Backprop with BCE
- Visualize hidden activations + decision boundary

### Milestone D — Grid-shape classifier
- MLP over flattened grid
- Weight templates per hidden neuron
- Activation bars + gradient maps

## 3) Minimal Backprop (for Dense layers)

Forward:
- z = W x + b
- a = f(z)

Backward:
- dL/dz = (dL/da) ⊙ f'(z)
- dL/dW = (dL/dz) xᵀ
- dL/db = dL/dz
- dL/dx = Wᵀ (dL/dz)

## 4) Build Order (fast feedback)

1) Loss functions + plots
2) Logistic neuron
3) 2-layer MLP (XOR)
4) Grid-shape MLP
5) Translation stress tests

If you choose a UI path (notebooks vs web), we can tailor the viz stack.
