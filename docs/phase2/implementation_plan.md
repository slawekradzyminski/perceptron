# Implementation Plan - Gradient Descent Lab (Phase 2)

Goal
Build a set of gradient descent modules that directly mirror Chapter 2 visuals
and exercises. The theme is that gradient descent works because it uses the
full gradient vector in the full parameter space, while low-dimensional slices
can be misleading.

Guiding principles
- Keep models small and explicit. Avoid autograd for core lessons.
- Show both analytic gradients and finite-difference gradients.
- Prefer clear visuals over performance. Favor correctness and pedagogy.
- Reuse existing LMS/MLP infrastructure where possible.

Scope (three layers)

Layer 1: Fully visible 2D bowl (Adaline / linear regression)
- Model: y_hat = w^T x + b for x in R^2
- Loss: 0.5 * (y - y_hat)^2
- Gradients:
  - dL/dw = -(y - y_hat) * x
  - dL/db = -(y - y_hat)
- UI:
  - Loss surface over (w1, w2) with GD path overlay
  - Gradient arrow at current point
  - Toggle: analytic gradient vs finite-difference gradient

Layer 2: Cross-entropy (binary + multi-class)
- Binary logistic neuron:
  - p = sigmoid(z), z = w^T x + b
  - L = -[y log p + (1-y) log(1-p)]
- Multi-class softmax:
  - p = softmax(z)
  - L = -log(p_k)
- UI:
  - Plot L1 and cross-entropy vs p(correct)
  - Show p(correct), loss, and loss delta per step
  - For sequences: average per-position cross-entropy

Layer 3: Loss landscape slices and the "wormhole" effect
- Snapshot parameter vector theta0
- Sample two random directions u, v (unit vectors)
- Define slice: theta(alpha, beta) = theta0 + alpha*u + beta*v
- Compute loss on a grid of (alpha, beta) values
- UI:
  - Contour plot for the slice
  - Recompute after a few GD steps to show how the slice changes
  - Optional: switch batches to show the loss surface shifting

Proposed backend changes
- Add a small gradient descent service under backend/services/gd_service.py
- Add routes under /gd for:
  - GET /gd/state: current theta, loss, gradient, step index
  - POST /gd/reset: init params, dataset, lr, mode
  - POST /gd/step: one GD step
  - POST /gd/slice: return loss grid for (alpha, beta) slice
- Add a lightweight model module:
  - backend/nn/adaline.py
  - backend/nn/logistic.py
  - backend/nn/softmax.py
  - backend/nn/finite_diff.py

Proposed frontend changes
- Add a new tab: "GD" (or reuse /lms with a sub-panel toggle)
- New panels:
  - Loss Surface (contour + path)
  - Gradient Compass (vector arrow)
  - Loss Comparison (L1 vs CE curve)
  - Slice Explorer (alpha/beta landscape)

Implementation steps
1) Backend math primitives
   - Implement losses, softmax, finite differences
   - Add unit tests for loss, softmax, finite-diff checks
2) Adaline GD state machine
   - Dataset: small 2D points with fixed targets
   - Step: compute grad, update, store trajectory
3) Loss surface endpoint
   - Compute loss over (w1, w2) grid
   - Return grid + trajectory
4) Frontend: 2D bowl demo
   - Plot contour and path
   - Draw gradient arrow
   - Toggle analytic vs numeric gradients
5) Cross-entropy panel
   - L1 vs CE plot
   - Show current p(correct) and loss values
6) Slice explorer
   - Sample u, v
   - Compute contour grid and update after steps
   - Optional batch switch

Acceptance criteria
- Users can explain why coordinate descent fails with a 2-parameter demo
- Users can see the difference between L1 and cross-entropy penalty curves
- Users can see a GD path moving downhill on a 2D surface
- Users can see a loss slice change after parameter updates
- Exercises 2.1 through 2.6 can be answered using the UI

Open questions
- Should the new features be a new route (/gd) or added to /lms?
- Do we want a simple static dataset, or reuse the existing custom dataset UI?
- How large should loss grids be to keep rendering smooth?
