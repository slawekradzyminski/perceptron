# Operations

## Start the app
- `scripts/restart_app.sh` (preferred)
  - Starts backend + frontend and checks health endpoints.

## Backend
- Install deps: `poetry install`
- Run API: `poetry run perceptron-api`
- Lint: `poetry run ruff check backend/`
- Lint + fix: `poetry run ruff check backend/ --fix`
- Format: `poetry run ruff format backend/`
- Run tests: `poetry run pytest`

## Frontend
- Install deps: `npm install` (from `frontend/`)
- Dev server: `npm run dev` (from `frontend/`)
- Lint: `npm run lint` (from `frontend/`)
- Lint + fix: `npm run lint:fix` (from `frontend/`)
- Build: `npm run build` (from `frontend/`)
- Tests: `npm test` (from `frontend/`)

## Stop processes
- `scripts/stop_app.sh`
