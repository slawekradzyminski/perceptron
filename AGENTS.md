# Agent Guide (Perceptron Visual Lab)

## Defaults
- Use project scripts proactively (don’t ask) for common tasks.
- Always run tests after code changes.
- Prefer small, incremental changes with clear verification.
- Always add tests for new code (both backend and frontend).

## Read First (required)
- Skim `docs/overview.md` to understand the core UX, routes, and terminology.
- Read `docs/features.md` and `docs/api.md` before changing UI or backend logic.
- Check `docs/operations.md` for expected workflows and scripts.

## Scripts (preferred)
- Restart app + health checks (no docker): `scripts/restart_app.sh`
- Restart app + health checks (docker): `scripts/restart_app.sh --docker`
- Verify app (start + health checks): `scripts/verify_app.sh`
- Stop app processes: `scripts/stop_app.sh`

## Backend
- Run API (Poetry): `poetry run perceptron-api`
- Run runner: `poetry run perceptron-runner --dataset or --epochs 10`
- Lint: `poetry run ruff check backend/`
- Lint + fix: `poetry run ruff check backend/ --fix`
- Format: `poetry run ruff format backend/`
- Backend tests: `poetry run pytest` (or `. .venv/bin/activate && pytest`)

## Frontend
- Dev server: `npm run dev` (from `frontend/`)
- Lint: `npm run lint` (from `frontend/`)
- Lint + fix: `npm run lint:fix` (from `frontend/`)
- Build: `npm run build` (from `frontend/`)
- Frontend tests: `npm test` (from `frontend/`)

## Conventions
- Backend is the single source of truth; frontend should not duplicate ML logic.
- Keep the UI consistent with backend outputs.
- If the app misbehaves, restart via `scripts/restart_app.sh`.

## Health checks
- API: `http://127.0.0.1:8000/state`
- Frontend: `http://127.0.0.1:5173/`

## After changes (required)
1) Run backend lint: `poetry run ruff check backend/`
2) Run backend tests: `poetry run pytest`
3) Run frontend lint: `npm run lint` (from `frontend/`)
4) Run frontend tests: `npm test` (from `frontend/`)
5) Ensure frontend build passes: `npm run build` (from `frontend/`)
6) If backend-related, restart app with `scripts/restart_app.sh`.
7) Frontend changes are hot-reloaded and do not require a restart.

## Lint as feedback loop
- **Always run lint after code changes** — it catches errors faster than tests or builds.
- Backend: `poetry run ruff check backend/` (use `--fix` to auto-fix)
- Frontend: `npm run lint` (use `lint:fix` to auto-fix)
- A clean lint run is a prerequisite before running tests or build.

Do not use Playwright MCP unless explicitly requested
