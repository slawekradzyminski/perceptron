# Implementation Plan

This plan targets a Python core with an optional TypeScript frontend. The goal is a didactic, step-by-step visualization of the perceptron learning rule.

## 0) Scope and learning goals
- Demonstrate perceptron updates on misclassified samples (hard threshold)
- Make linear separability obvious (OR vs XOR)
- Visualize weights as "dials" and show per-cell contributions
- Show convergence behavior (mistakes per epoch) and failure modes

## 1) Core Python model (backend)
**Deliverables**
- `backend/perceptron.py`: `Perceptron` class
  - `predict_score(x)`, `predict_label(x)`
  - `train_step(x, y, lr)` -> returns diagnostics
- `backend/datasets.py`
  - 2D boolean datasets (OR / AND / XOR)
  - Grid-shape datasets with translations
- `backend/metrics.py`
  - mistakes, accuracy, margin tracking

**Notes**
- Use labels in {-1, +1}
- Update only on mistakes: `w += lr * y * x`, `b += lr * y`

## 2) Visualization primitives (Python first)
**Deliverables**
- `backend/viz_2d.py`
  - scatter plot of points
  - decision boundary line from `w1*x + w2*y + b = 0`
  - optional animation frame generator
- `backend/viz_grid.py`
  - render input grid
  - render weight grid
  - render contribution grid (x ⊙ w)
  - score meter

This enables a notebook-driven prototype before wiring to the web UI.

## 3) Frontend (TypeScript, optional but recommended)
**Suggested stack**
- Vite + React + TypeScript
- Canvas or SVG for grid + line plotting

**Data flow**
- Frontend holds UI state (current sample, play/pause, speed)
- Backend exposes a small API:
  - `POST /reset`
  - `POST /step` -> returns updated weights, score, mistake flag
  - `GET /state`

**Alternative**
- Skip HTTP and run entirely client-side (TS ports of perceptron logic)
- Keep Python as reference implementation + validation

## 4) Training UX
**Controls**
- Dataset selector (OR / AND / XOR / shapes)
- Learning rate slider
- Step / Play / Reset
- Toggle: show contributions grid
- Toggle: train on translations

**Visual feedback**
- Highlight current sample
- Animate weight updates per step
- Mistakes per epoch chart

## 5) Milestone sequence
1. Python-only 2D perceptron + boundary plot
2. Grid input + weights + contributions
3. Translation dataset + convergence chart
4. TS frontend + API integration
5. Optional Adaline/LMS comparison

## 6) Verification checklist
- OR and AND converge quickly
- XOR never fully converges
- Translational training generalizes across positions
- UI reflects weight updates per mistake
