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
- **Attention Patterns (Glass Box - Chapter 8)**
  - Visualize attention weights for any input text.
  - Select specific layers and heads to explore.
  - Heatmap shows which tokens attend to which.
  - Educational notes explain head patterns (previous token, induction, etc.).
- **Logit Lens (Glass Box - Chapter 7)**
  - See what the model "believes" at each layer.
  - Apply output layer to intermediate hidden states.
  - Watch predictions evolve from uncertain to confident.
  - Table shows top predictions with probabilities per layer.
- **KV Cache Calculator (Chapter 8)**
  - Compare memory usage across attention variants.
  - Supports MHA, MQA, GQA, and MLA (DeepSeek R1).
  - Interactive controls for model parameters.
  - Demonstrates why MLA enables longer context windows.
  - Preset configurations for GPT-2, Llama 3, DeepSeek R1.

## Convolution Lab ("/convolution")
- **Interactive Convolution Visualization**
  - See how a kernel slides across an input image.
  - Hover over activation map cells to highlight the receptive field.
  - Same kernel weights used everywhere (weight sharing visualization).
- **Kernel Editor**
  - Choose from preset kernels: identity, edge, sobel, blur, sharpen, emboss.
  - Manually edit kernel weights in a 3×3 grid.
  - See immediate effect on the activation map.
- **Sample Images**
  - Preset patterns: gradient, checkerboard, cross, edges, diagonal.
  - Upload custom images (resized to current size).
  - Adjust image size from 4×4 to 16×16.
- **Step-by-Step Math**
  - For any hovered position, displays:
    - The extracted patch (receptive field)
    - The kernel weights
    - Element-wise products
    - Final sum (activation value)
- **Convolution Parameters**
  - Padding: 0, 1, or 2 pixels of zero-padding.
  - Stride: 1, 2, or 3 pixel step size.
- **Digit Recognition**
  - Upload handwritten digit images (0-9).
  - See CNN predictions with confidence scores.
  - Visualize layer activations (Conv1 → Pool → Conv2 → Pool).
  - Probability distribution across all 10 digit classes.
  - Uses pretrained model with ~97% accuracy on MNIST.
