# Implementation Plan — UI for Error Surfaces + Multi‑Layer Network

Last updated: 2026-01-02

## Goal
Expose the backend error/loss and multi‑layer network capabilities in the UI with clear, testable panels and deterministic data flows.

## Phase 1 — API + Data Contracts
- Add backend endpoints that expose:
  - Error surface grid (MSE/BCE) for a chosen model and fixed dataset.
  - MLP internals (layer weights, activations, gradients, decision boundary).
- Define frontend TypeScript types mirroring response payloads.
- Add unit tests for endpoint outputs and schema validation.

## Phase 2 — UI Panels
- Add a new “Error Surface” panel:
  - Model selector, loss selector, grid resolution.
  - Heatmap rendering + legend.
- Add a new “MLP Internals” panel:
  - Layer tabs (weights, activations, gradients).
  - Clear labels for pre‑update vs post‑update.
- Wire panels to API via a dedicated hook.
- Add RTL tests for panel rendering and state transitions.

## Phase 3 — UX + Layout Integration
- Integrate panels into the main layout without breaking current perceptron flow.
- Add loading/empty/error states.
- Ensure keyboard shortcuts and reset flow still work.
- Add Playwright MCP click‑through test script for the new UI.

## Phase 4 — Polish + Documentation
- Add legend/explanation text for error surfaces and gradients.
- Document endpoints in `context.md`.
- Update `ERROR_MULTI_LAYERS.md` with UI entry points and screenshots.

## Progress Tracker
- [x] Phase 1 — API + Data Contracts
- [x] Phase 2 — UI Panels
- [x] Phase 3 — UX + Layout Integration
- [x] Phase 4 — Polish + Documentation

## Testing
- Backend: pytest for endpoints + math.
- Frontend: RTL for panels + hooks.
- E2E: Playwright MCP click‑through for main flows and new panels.
