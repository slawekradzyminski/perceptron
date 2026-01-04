# Phase 3 Progress

## Status (2026-01-04)
- Reformatted Phase 3 docs for clarity.
- Added TinyGPS city coordinate fixtures (Madrid, Paris, Berlin, Barcelona).
- Added TinyGPS datasets: `madrid-paris-berlin` (1D lon) and `four-cities` (2D lat/lon).
- Restored 2-city TinyGPS datasets (`paris-berlin`, `paris-madrid`) for exercise tables.
- Implemented TinyGPS math helpers (forward/backward for 1D and 2D softmax-CE).
- Implemented single-neuron regression helpers (MSE + L1).
- Added Backprop service + `/backprop` API routes (state, tinygps reset/step, regression reset/step).
- Added backend tests for TinyGPS/regression math, datasets, and new API routes.
- Backend Phase 3 scope complete (TinyGPS + regression + API wiring).
- Frontend split into `/backprop/tinygps` and `/backprop/regression` pages with dedicated components and CSS.
- Restored 2-city TinyGPS datasets for book exercises.
- Added regression plot (step 0/1/7 overlays) and sample scatter in Regression page.
- Regression API state now returns full sample list for plotting.
- Map of Language scaffold added under `scripts/map_of_language/`.
- Added Map of Language placeholder doc at `docs/phase3/map_of_language.md`.

## Next up
- Confirm dataset points/rounding for exercise tables; align tests to book values.
- Confirm regression dataset points (currently defaulting to a simple line: y = 2x + 1 with x=1..4).
