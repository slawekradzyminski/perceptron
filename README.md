# Perceptron Visual Lab

Interactive, code-first exploration of the classic perceptron: how the decision boundary moves, why XOR fails, and how weight updates work under the hood. The project favors transparency over abstraction.

## Repo layout (planned)
- `backend/`: Python core (perceptron model, datasets, training loop, metrics)
- `frontend/`: TypeScript UI (2D plot + grid/weight-board visualizer)
- `docs/`: design notes, experiments, and references

## Milestones (planned)
1. 2D perceptron with live boundary animation
2. Grid/weight-board visualization (input, weights, contributions, score meter)
3. Training on translated patterns and mistake tracking
4. (Optional) Adaline/LMS comparison for gradient descent intuition

## Notes
- The repo is intentionally minimal right now. See `PLAN.md` for the detailed implementation plan.

## Backend (Poetry)
- Install: `poetry install`
- Run API: `poetry run perceptron-api`
- Run runner: `poetry run perceptron-runner --dataset or --epochs 10`
