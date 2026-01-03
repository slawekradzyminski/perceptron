# Feature Guide

This file summarizes the user-facing features and how to interpret them.

## Main Lab ("/" route)
- **Switchboard**
  - Input (x): current sample as a grid.
  - Weights (before): weights used to compute the score for this step.
  - Contribution (pre-update): elementwise product x ⊙ w.
  - Bias (before): scalar contribution to the score.
  - Score (s): sum(x ⊙ w) + b using pre-update weights.
- **After Update panel**
  - Shows weights and bias after the perceptron update is applied.
- **Step Math**
  - Shows the numeric calculation for s, the prediction, the update rule, and the reason an update was or was not applied.

## Custom Dataset (modal)
- Choose rows/cols (1–5) and define samples.
- Each sample is a grid of -1/+1 values plus a label y (-1 or +1).
- Applying the dataset resets the model and reuses backend logic.

## Diagnostics ("/diagnostics")
- **Error Surface**
  - Visualizes mean squared error over a grid of (w1, w2) values.
  - Only available for 2D inputs (grid_rows * grid_cols == 2).
  - Hovering a cell shows the corresponding w1/w2 values and loss.
- **MLP Internals**
  - Shows a 1-hidden-layer network: hidden templates, gradients, and post-update weights.
  - Makes backprop signals visible on the same grid representation.

## LMS ("/lms")
- A step-by-step LMS table with gradients and weight updates.
- Shows up to 16 steps in the table for readability.
- Includes a mini error trend sparkline for intuition.
