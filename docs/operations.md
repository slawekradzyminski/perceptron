# Operations

## Start the app
- `scripts/restart_app.sh` (preferred)
  - Starts backend + frontend and checks health endpoints.

## Backend
- Install deps: `poetry install`
- Run API: `poetry run perceptron-api`
- Run tests: `poetry run pytest`

## Frontend
- Install deps: `npm install` (from `frontend/`)
- Dev server: `npm run dev` (from `frontend/`)
- Tests: `npm test` (from `frontend/`)

## Stop processes
- `scripts/stop_app.sh`
