# Perceptron Visual Lab — Overview

This project is an educational, step-by-step visualization of perceptron learning and related gradient methods. It combines a FastAPI backend with a React frontend to make each update and calculation explicit.

## Routes
- `/` Main lab view: perceptron step-by-step learning with the switchboard panels, score, and step math.
- `/diagnostics` Diagnostics: Error Surface + MLP Internals.
- `/lms` LMS view: least-mean-squares update table with gradients.
- `/mlp` MLP Trainer: 2-layer MLP with hidden templates.
- `/gd` Gradient Descent: loss function comparisons.
- `/backprop/tinygps` TinyGPS: multi-class classification.
- `/backprop/regression` Regression: line fitting with backprop.
- `/deep` Deep Learning Lab: variable-depth networks and region counting.
- `/alexnet` AlexNet Explorer: visualize CNN filters and activation maps.
- `/transformer` Transformer Flow: tokenization, block trace, and model scale comparison.
- `/convolution` Convolution Lab: interactive 2D convolution visualization with step-by-step math.

## Keyboard shortcuts
- `S`: Step
- `R`: Reset

## Datasets
- `or` / `xor`: 1x2 input grid (two-switch visualization).
- `custom`: user-defined grid size (1–5 per side) with user-provided samples.

## Backend-first model
All ML logic is computed on the backend. The frontend is a renderer and controller that mirrors the API state.
