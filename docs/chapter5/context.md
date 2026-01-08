# Chapter 5 Context (AlexNet & Architectural Scale)

## Overview

Chapter 5 marks the transition from small, explainable models to large-scale architectures that "see" and "process" the world through high-dimensional embedding spaces. It focuses on AlexNet (2012) as the tipping point for Deep Learning and contrasts it with modern Transformers (ChatGPT).

## Core Concepts

### 1. Convolutional Neural Networks (CNNs)

- **Kernels/Filters**: Small tensors of learned weights (e.g., 11×11×3) that slide over an image to compute dot products.
- **Pattern Detectors**: Early layers learn basic features like edge detectors and color blobs.
- **Feature Hierarchy**: Deeper layers combine low-level features into abstract concepts, such as corner detectors (Layer 2) or face/citrus detectors (Layer 5).
- **Activation Maps**: The 2D results of sliding a kernel over an input, showing where specific patterns occur.

### 2. Transformer Architecture (The "T" in GPT)

- **Tokenization**: Breaking text into word fragments (tokens).
- **Embedding**: Converting tokens into high-dimensional vectors.
- **Compute Blocks**: Stacking transformer blocks that perform matrix operations, maintaining matrix shape throughout the stack.
- **Autoregressive Generation**: Predicting the next token using only the last column of the final output matrix, then appending it to the input for the next cycle.

### 3. Embedding/Latent Spaces

- **Semantic Meaning**: Vectors for similar concepts (e.g., elephants) cluster together in 4096-dimensional space.
- **Vector Arithmetic**: Directionality in space is meaningful (e.g., "King - Man + Woman ≈ Queen").
- **Activation Atlases**: Low-dimensional projections (UMAP/t-SNE) of high-dimensional activations to see how a model organizes the world.

## What the Lab Needs

To match the chapter's scale, the lab needs tools to peek "under the hood" of large models:

- **AlexNet Explorer**: Visualizing first-layer kernels and activation maps for uploaded images.
- **Transformer Flow**: A step-by-step visualization of how a prompt becomes a token matrix and eventually a new word.
- **Latent Space Visualizer**: 2D projections of embeddings to show clustering behavior.

## Exercise Mapping

- **5.3 - 5.4**: Understanding input/output dimensions and why early layers are easier to visualize.
- **5.5 - 5.6**: Identifying specialized detectors (corners, faces) in hidden layers.
- **5.8**: Explaining Activation Atlases.
- **5.10**: Comparing parameter counts (AlexNet 60M vs GPT-4 1T+).
