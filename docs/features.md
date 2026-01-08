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

## Deep Learning Lab ("/deep")
- **Architecture Controls**
  - Depth: Number of hidden layers (1-6).
  - Width: Neurons per hidden layer (4-64).
  - Dataset: circles, spiral, xor, or baarle (Baarle-Hertog enclaves).
- **Visualization**
  - Decision Boundary: Shows predicted class regions.
  - Region Tiling: Colors pixels by unique linear region (activation pattern).
- **Region Analysis**
  - Actual Regions: Count of unique ReLU activation signatures.
  - Theoretical Max: Upper bound from Montufar et al. formula.
  - Efficiency: ratio of actual to theoretical regions.
- **Comparison Table**
  - Collect results from different architectures for Exercise 4.9.
  - Compares depth vs width trade-offs for region complexity.

## AlexNet Explorer ("/alexnet")
- **Layer Navigation**
  - Select layers 1-5 to view different convolutional layers.
  - Layer 1: Edge and color blob detectors (11×11×3 RGB filters).
  - Layer 2: Corner and texture detectors.
  - Layers 3-5: Higher-level abstract feature detectors.
- **Filter Grid**
  - Displays the 64 first-layer filters as a visual 8×8 grid.
  - Hover to highlight corresponding activation maps.
- **Test Images**
  - Pre-loaded samples: gradient, checkerboard, noise.
  - Upload custom images to see filter activations.
- **Activation Maps**
  - Shows where each filter fires on the input image.
  - Brighter = stronger activation.
- **Model Info**
  - AlexNet has ~61M parameters across 5 convolutional layers.

## Transformer Flow ("/transformer")
- **Tokenization**
  - Enter text and see it split into color-coded tokens.
  - Displays token IDs from the cl100k_base encoding.
  - Shows the embedding matrix shape (tokens × 768 dimensions).
- **Block Trace**
  - Visualize how the input matrix flows through transformer blocks.
  - Shows input/output shapes remain constant through all blocks.
  - Demonstrates that only the last column determines the next token.
- **Generation Simulation**
  - Step through autoregressive generation one token at a time.
  - See how each predicted token extends the sequence.
  - Key insight: the last column of the output determines the next word.
- **Model Scale Comparison**
  - Compare models from LeNet-5 (60K params) to GPT-4 (1.8T params).
  - Interactive comparison showing parameter ratios.
  - Growth chart showing exponential increase in model sizes.
  - Full table with all historical models (1998-2023).
