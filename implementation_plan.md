# Implementation Plan (Living)

Last updated: 2026-01-01

## Order of Implementation
1. Python core: perceptron math + datasets + metrics
2. Python visualization helpers (2D + grid)
3. Training runner / CLI or notebook harness
4. TypeScript frontend (optional): interactive UI + plots
5. Backend API (optional): step/reset endpoints
6. Optional: Adaline/LMS comparison

## Progress Tracker
- [x] 1. Perceptron core (Python)
  - [x] `Perceptron` class with `predict_score`, `predict_label`, `train_step`
  - [x] Unit-friendly deterministic init (seed)
  - [x] Pytest coverage for core behaviors
- [x] 2. Datasets (Python)
  - [x] OR / AND / XOR datasets in {-1,+1}
  - [x] Grid shape dataset generator
  - [x] Translation generator
- [x] 3. Metrics (Python)
  - [x] Mistake count per epoch
  - [x] Accuracy helpers
  - [x] Margin tracking
- [x] 4. 2D Visualization helpers (Python)
  - [x] Boundary math helpers
  - [x] Optional matplotlib plotting helper
- [x] 5. Grid Visualization helpers (Python)
  - [x] Contribution + score math
  - [x] Optional matplotlib grid rendering
- [x] 6. Runner / demo harness
  - [x] Minimal CLI runner (`backend/runner.py`)
  - [x] Tested training stats output
- [~] 7. TypeScript frontend (optional)
  - [x] Vite + TS skeleton
  - [x] Initial UI scaffold
  - [x] RTL tests (Vitest + @testing-library/dom)
  - [x] Client-side perceptron updates
  - [x] Optional API toggle wired
- [x] 8. API glue (optional)
  - [x] `/reset`, `/step`, `/state` endpoints (stdlib server)
  - [x] Frontend integration
- [ ] 9. Optional Adaline/LMS
  - [ ] LMS model + loss surface demo

## Notes
- Labels are {-1, +1}.
- Updates only on mistakes: w += lr * y * x; b += lr * y.
- Keep everything small, inspectable, and deterministic.
