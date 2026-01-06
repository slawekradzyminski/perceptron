import "@testing-library/jest-dom/vitest";
import React, { type ReactNode } from "react";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// Mock react-leaflet for tests (doesn't work in jsdom)
vi.mock("react-leaflet", () => {
  return {
    MapContainer: ({ children }: { children: ReactNode }) =>
      React.createElement("div", { "data-testid": "mock-map" }, children),
    TileLayer: () => null,
    CircleMarker: ({ children }: { children: ReactNode }) =>
      React.createElement("div", { "data-testid": "mock-marker" }, children),
    Polyline: ({ children }: { children: ReactNode }) =>
      React.createElement("div", { "data-testid": "mock-polyline" }, children),
    Polygon: ({ children }: { children: ReactNode }) =>
      React.createElement("div", { "data-testid": "mock-polygon" }, children),
    Rectangle: () => React.createElement("div", { "data-testid": "mock-rectangle" }),
    Tooltip: ({ children }: { children: ReactNode }) =>
      React.createElement("div", { "data-testid": "mock-tooltip" }, children),
    useMap: () => ({
      fitBounds: () => {},
    }),
  };
});

HTMLCanvasElement.prototype.getContext = (() =>
  ({
    canvas: { width: 100, height: 50 },
    clearRect: () => undefined,
    fillRect: () => undefined,
    beginPath: () => undefined,
    moveTo: () => undefined,
    lineTo: () => undefined,
    stroke: () => undefined,
    arc: () => undefined,
    fill: () => undefined,
    strokeRect: () => undefined,
    setLineDash: () => undefined,
    set fillStyle(_: string) {},
    set strokeStyle(_: string) {},
    set lineWidth(_: number) {},
  }) as unknown) as unknown as typeof HTMLCanvasElement.prototype.getContext;
