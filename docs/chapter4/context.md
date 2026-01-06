# Chapter 4 Context (Deep Learning Lab)

## Overview
Chapter 4 shifts focus from *how* a single neuron learns (backprop) to *what* deep networks actually represent: **geometric folding**. It addresses the "Universal Approximation Theorem" (width) versus the "Efficiency of Depth" (layers).

The core intuition is that while a wide two-layer network *can* theoretically represent any function (like the complex Baarle-Hertog border), it is inefficient and hard to train. Deep networks, by contrast, recursively "fold" the input space, creating exponentially more decision regions with fewer parameters.

## Implementation Status

✅ **Completed:**
- `DeepMlp` model with variable depth/width and ReLU activations
- Region counting via activation signature analysis
- Baarle-Hertog dataset loader (with synthetic fallback)
- `/deep` API endpoints (reset, step, epoch, regions, boundary, comparison)
- `DeepPage.tsx` frontend with tiling visualization
- Architecture controls (depth, width, learning rate, seed)
- Comparison table for Exercise 4.9

**Route:** `/deep`

## Core Concepts (The Geometry of Depth)
- **ReLU as Folding**: A single ReLU neuron folds the input space along a line (in 2D inputs). It effectively creates a "hinge."
- **Recursive Partitioning**:
  - **Layer 1**: Cuts the map into basic regions (half-planes folded up).
  - **Layer 2**: Folds the *already folded* space from Layer 1. This allows a single neuron in Layer 2 to create multiple "joints" in the original input space.
  - **Layer N**: Continues this recursion. The complexity of the "tiling" (the number of distinct linear regions) grows exponentially with depth but only polynomially with width.
- **The Baarle-Hertog Problem**: A real-world geographical oddity (Belgium/Netherlands enclaves) used as the stress test. Its border is so fragmented that simple plane-cutting models fail, but deep recursive folding succeeds.

## Key Formulas
**Theoretical Maximum Regions ($N_r$)**
For a network with input dimension $D_{in}$, width $D$, and $K$ hidden layers:
$$N_r = \left(\frac{D}{D_{in}} + 1\right)^{D_{in}(K-1)} \left(\frac{D^2 + D + 2}{2}\right)$$
*Note: This formula (from Montufar et al. / Prince) is an upper bound. The lab will compare this theoretical max against the "Actual Regions" found by the trained model.*

## What the Lab Needs
To match the chapter's narrative and exercises, the lab requires:

### 1. Flexible Architecture (Depth vs. Width)
- The fixed 2-layer MLP (`MlpTwoLayer`) from Chapter 3 is insufficient.
- We need a generic `DeepMlp` that supports $K$ layers of width $D$.

### 2. The Baarle-Hertog Dataset
- A 2D coordinate dataset representing the complex enclaves.
- The chapter provides code to load this from an image mask; we should replicate this loader or use pre-computed points.

### 3. Region Counting & Visualization
- **Region Map**: Instead of just showing the decision boundary ($p=0.5$), we need to visualize the *tiling* of the input space.
- **Activation Patterns**: Two points belong to the same "linear region" if they trigger the exact same set of ReLU activations (active/inactive) across all neurons.
- **Counter**: A metric that counts how many unique regions the current model has carved the space into.

## Exercise Mapping
- **4.1 - 4.6 (Conceptual)**: Addressed via text/Q&A panels.
- **4.7 - 4.8 (Tracing Points)**: Manual calculation checks.
- **4.9 (Architecture Comparison)**: The core interactive task. "Train a 2-layer vs. 3-layer vs. 5-layer network and record the number of regions."
- **4.10 (Theory vs. Practice)**: Compare the counted regions to the formula above.
- **4.12 (Challenge)**: Solve Baarle-Hertog with <100 neurons.