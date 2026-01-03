# Perceptron Visual Lab — Overview

This project is an educational, step-by-step visualization of perceptron learning and related gradient methods. It combines a FastAPI backend with a React frontend to make each update and calculation explicit.

## Routes
- `/` Main lab view: perceptron step-by-step learning with the switchboard panels, score, and step math.
- `/diagnostics` Diagnostics: Error Surface + MLP Internals.
- `/lms` LMS view: least-mean-squares update table with gradients.

## Keyboard shortcuts
- `S`: Step
- `R`: Reset

## Datasets
- `or` / `xor`: 1x2 input grid (two-switch visualization).
- `custom`: user-defined grid size (1–5 per side) with user-provided samples.

## Backend-first model
All ML logic is computed on the backend. The frontend is a renderer and controller that mirrors the API state.
