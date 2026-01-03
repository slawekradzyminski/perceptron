# Playwright MCP Smoke Test — Diagnostics Panels

Goal: verify Error Surface and MLP Internals panels render and react to controls.

## Steps
1) Navigate to the frontend (`http://127.0.0.1:5173/`).
2) Confirm the “Error Surface” panel shows a heatmap.
3) Change “Steps” to 10 and verify the heatmap still renders.
4) Confirm the “MLP Internals” panel shows hidden templates.
5) Change “Hidden units” to 3 and verify templates update.
6) Switch Dataset to “Custom”.
7) Open the modal, add a sample, and click “Apply custom dataset”.
8) Confirm diagnostics panels show data for custom dataset.
9) Switch Dataset back to “OR” and confirm diagnostics panels reload.

Notes: if API errors appear, run `scripts/restart_app.sh` and retry.
