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

## What the lab provides (to match Chapter 3 + exercises)

### A) TinyGPS softmax classifier (main backprop walkthrough)
The lab reproduces the simplified longitude-only model used in the book:
- Softmax + cross-entropy with explicit `h`, `ŷ`, and gradients.
- Exercise presets for Paris–Berlin (4) and Paris–Madrid (4).
- Book Table with per-cell hover explanations.
- Loss trend chart and a live backprop flow diagram.
- City map with the 1D decision boundary (vertical longitude line).

### B) Regression with MSE (Exercises 3.18–3.23)
The regression page includes:
- Dataset editor + book presets.
- Plot with line overlays (step 0/1/7) and scatter points.
- Book Table Mode with book-style rounding.

### C) Regression with L1 (Exercises 3.24–3.29)
Same UI as MSE, with L1 gradients and the optional “book-literal” loss fix for the known table discrepancy.

### D) Optional: Map of Language (Exercise 3.30)
The supporting demo remains in-repo and is documented at `docs/phase3/map_of_language.md`.

## Practical “do I have enough already?” check
From the repo snapshot, you already have:
- softmax, CE helpers, finite differences, activations, losses
- MLP primitives and API routes for GD/MLP/LMS

So for Chapter 3, the core exercises are covered in the UI. Remaining work is optional polish (e.g., 2D boundary map for `four-cities`).

That aligns almost one-to-one with the Chapter 3 exercise blocks.

If you want, paste (or upload) the pages containing the exercise tables for 3.4–3.17 (the step-by-step parameter table). Then I can give you a Codex-ready spec that matches the table columns exactly (what to compute and where rounding matters), so you can verify your outputs line-by-line.
