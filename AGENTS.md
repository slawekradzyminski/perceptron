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
- Restart app + health checks (ollama only in docker): `scripts/restart_app.sh --ollama`
- Restart app + health checks (full docker stack): `scripts/restart_app.sh --docker`
- Verify app (start + health checks): `scripts/verify_app.sh`
- Stop app processes: `scripts/stop_app.sh`

## Backend
- Run API (Poetry): `poetry run perceptron-api`
- Run runner: `poetry run perceptron-runner --dataset or --epochs 10`
- **Full check (lint + types + tests)**: `poetry run check` (~2-3 min)
- Lint only: `poetry run ruff check backend/`
- Lint + fix: `poetry run ruff check backend/ --fix`
- Format: `poetry run ruff format backend/`
- Type check only: `poetry run mypy backend/`
- Tests only: `poetry run pytest`

## Frontend
- Dev server: `npm run dev` (from `frontend/`)
- **Full check (lint + build + tests)**: `npm run check` (~30s)
- Lint only: `npm run lint`
- Lint + fix: `npm run lint:fix`
- Build only: `npm run build`
- Tests only: `npm test`

## Conventions
- Backend is the single source of truth; frontend should not duplicate ML logic.
- Keep the UI consistent with backend outputs.
- If the app misbehaves, restart via `scripts/restart_app.sh`.

## Health checks
- API: `http://127.0.0.1:8000/state`
- Frontend: `http://127.0.0.1:5173/`

## After changes (required)
1) **Backend**: `poetry run check` (~2-3 min) - runs lint, type check, and tests
2) **Frontend**: `npm run check` (~30s from `frontend/`) - runs lint, build, and tests
3) If backend-related, restart app with `scripts/restart_app.sh`.
4) Frontend changes are hot-reloaded and do not require a restart.

## Quick feedback loop
- For fast iteration, use individual commands:
  - Backend lint: `poetry run ruff check backend/` (use `--fix` to auto-fix)
  - Frontend lint: `npm run lint` (use `lint:fix` to auto-fix)
- Before committing, always run the full `check` commands.

Do not use Playwright MCP unless explicitly requested
