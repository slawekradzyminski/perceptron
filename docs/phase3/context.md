# Chapter 3 Context (Backprop Lab)

## Overview
Chapter 3 is where you stop “nudging weights” and start doing full backprop via the chain rule, with softmax + cross-entropy as the key worked example (the TinyGPS city classifier). Then you practice the same machinery on single-neuron regression with MSE and L1.

## What you need conceptually (minimum math)
- **Computational graph + chain rule**: treat the model as blocks and multiply derivatives block-by-block instead of differentiating the whole thing at once. The chapter explicitly contrasts the “clunky full derivative” with the modular chain rule approach.
- **Softmax + cross-entropy shortcut**: the big simplification the chapter leans on is:

```text
∂L/∂h = y^ - y
```

The chapter explains the log and exp “basically cancel out.”

- **Linear neuron partials**: for a neuron `h_k = m_k * x + b_k`:

```text
∂h_k/∂m_k = x
∂h_k/∂b_k = 1
```

The text walks through this intuition (“either way it’s a line”) and then plugs it into the final derivative.

From those, your core training gradients for the TinyGPS classifier become:

```text
∂L/∂m_k = x * (y^_k - y_k)
∂L/∂b_k = (y^_k - y_k)
```

## What you need to implement (to match Chapter 3 + exercises)

### A) TinyGPS softmax classifier (main backprop walkthrough)
This is the model from the early pages: one input (longitude `x`), 2–3 output neurons (cities), softmax, cross-entropy.

Implement these functions (keep them tiny and explicit):

```text
forward(x, m: [K], b: [K]) -> (h: [K], p: [K])
  h[k] = m[k]*x + b[k]
  p = softmax(h)

loss_ce(p, y_index) -> float
  L = -log(p[y_index])

backward(x, p, y_onehot) -> grads
  dL_dh = p - y_onehot
  dL_dm[k] = x * dL_dh[k]
  dL_db[k] = dL_dh[k]

sgd_step(m, b, grads, lr) -> (m_new, b_new)
  update in the opposite direction of gradients
```

**What this unlocks in the exercises**
- Filling the “forward pass” table and gradients for the simplified two-city model (Paris vs Berlin / Paris vs Madrid) relies exactly on these computations.
- Then you apply gradient descent with `lr=0.1` and check loss/accuracy per step.

**Strong recommendation**: add a finite-difference gradient check for one step (pick a parameter `m_k`, perturb it ±ε, compare numerical slope to your `dL_dm[k]`). This will make the chain rule feel “proved” in code.

**Good news**: your repo already has softmax, losses, and finite-diff plumbing in `backend/nn/...`, so this is mostly wiring + a “TinyGPS” model wrapper.

### B) Single neuron regression with squared error (MSE) (Exercises 3.18–3.23)
The exercises switch to one neuron `y^ = m * x + b` trained by backprop/gradient descent on simple points.

Implement:

```text
y_hat = m*x + b
L = 0.5*(y_hat - y)^2

dL_dm = (y_hat - y)*x
dL_db = (y_hat - y)

SGD updates with lr=0.1
```

**Visual to add (worth it)**
- Plot scatter points and the learned line at steps 0, 1, 7.
- Measure “steps to get within ±0.001” of true `m,b` and see LR trade-offs (explicitly asked).

### C) Single neuron regression with L1 loss (Exercises 3.24–3.29)
Same setup, but L1 loss.

Implement:

```text
L = abs(y_hat - y)

subgradient:
  dL_dyhat = +1 if (y_hat - y) > 0
             -1 if (y_hat - y) < 0
              0 if exactly 0

then:
  dL_dm = dL_dyhat * x
  dL_db = dL_dyhat
```

Then compare convergence vs MSE (explicitly asked).

### D) Optional: “Map of Language” / UMAP cluster demo (Exercise 3.30)
The chapter describes projecting high-dimensional representations down to 2D with UMAP to see clustering, and the exercises point you to the supporting code repo to run an interactive “map of language.”

If you want to do this:
- extract hidden states / embeddings from a model layer
- run UMAP to 2D
- plot points + inspect clusters by layer

This is optional compared to the core backprop exercises.

## Practical “do I have enough already?” check
From the repo snapshot, you already have:
- softmax, CE helpers, finite differences, activations, losses
- MLP primitives and API routes for GD/MLP/LMS

So for Chapter 3, it’s mainly:
- add a TinyGPS softmax classifier module + step-by-step table reproduction outputs
- add two regression trainers: MSE vs L1 + plotting/step counting

That aligns almost one-to-one with the Chapter 3 exercise blocks.

If you want, paste (or upload) the pages containing the exercise tables for 3.4–3.17 (the step-by-step parameter table). Then I can give you a Codex-ready spec that matches the table columns exactly (what to compute and where rounding matters), so you can verify your outputs line-by-line.
