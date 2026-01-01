# Agent Guide (Perceptron Visual Lab)

## Defaults
- Use project scripts proactively (don’t ask) for common tasks.
- Always run tests after code changes.
- Prefer small, incremental changes with clear verification.

## Scripts (preferred)
- Restart app + health checks: `scripts/restart_app.sh`
- Verify app (start + health checks): `scripts/verify_app.sh`
- Stop app processes: `scripts/stop_app.sh`

## Backend
- Run API (Poetry): `poetry run perceptron-api`
- Run runner: `poetry run perceptron-runner --dataset or --epochs 10`
- Backend tests: `poetry run pytest` (or `. .venv/bin/activate && pytest`)

## Frontend
- Dev server: `npm run dev` (from `frontend/`)
- Frontend tests: `npm test` (from `frontend/`)

## Conventions
- Backend is the single source of truth; frontend should not duplicate ML logic.
- Keep the UI consistent with backend outputs.
- If the app misbehaves, restart via `scripts/restart_app.sh`.

## Health checks
- API: `http://127.0.0.1:8000/state`
- Frontend: `http://127.0.0.1:5173/`

## After changes (required)
1) Run backend tests.
2) Run frontend tests.
3) If UI-related, restart app with `scripts/restart_app.sh`.
