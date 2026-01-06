# Perceptron Visual Lab

Interactive, code-first exploration of the classic perceptron: how the decision boundary moves, why XOR fails, and how weight updates work under the hood. The project favors transparency over abstraction.

## Repo layout
- `backend/`: Python core (perceptron model, datasets, training loop, metrics)
- `frontend/`: TypeScript UI (2D plot + grid/weight-board visualizer)
- `docs/`: feature notes, API docs, and operational guides

## Docs
- Overview: `docs/overview.md`
- Feature guide: `docs/features.md`
- Backend API: `docs/api.md`
- Operations: `docs/operations.md`
- Chapter 2: `docs/chapter2/`
- Chapter 3: `docs/chapter3/`
- Chapter 4: `docs/chapter4/`

## Backend (Poetry)
- Install: `poetry install`
- Run API: `poetry run perceptron-api`
- Run runner: `poetry run perceptron-runner --dataset or --epochs 10`

## Frontend
- Install: `npm install` (from `frontend/`)
- Dev server: `npm run dev` (from `frontend/`)
- Tests: `npm test` (from `frontend/`)
