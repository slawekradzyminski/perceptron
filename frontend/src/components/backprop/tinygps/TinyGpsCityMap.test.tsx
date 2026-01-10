import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import type { ReactNode } from "react";
import { TinyGpsCityMap } from "./TinyGpsCityMap";
import type { TinyGpsState } from "../../../types";

vi.mock("react-leaflet", () => ({
  MapContainer: ({ children }: { children: ReactNode }) => (
    <div data-testid="map">{children}</div>
  ),
  TileLayer: () => <div data-testid="tile" />,
  CircleMarker: ({ children }: { children: ReactNode }) => (
    <div data-testid="marker">{children}</div>
  ),
  Polyline: ({ children }: { children?: ReactNode }) => (
    <div data-testid="polyline">{children}</div>
  ),
  Polygon: ({ children }: { children?: ReactNode }) => (
    <div data-testid="polygon">{children}</div>
  ),
  Tooltip: ({ children }: { children: ReactNode }) => (
    <div data-testid="tooltip">{children}</div>
  ),
  useMap: () => ({ fitBounds: vi.fn() }),
}));

const mockState: TinyGpsState = {
  dataset: "europe",
  mode: "1d",
  cities: ["paris", "madrid"],
  idx: 0,
  lr: 0.1,
  sample_count: 2,
  feature_order: ["lon"],
  sample_order: [0, 1],
  params: { m: [0.1, -0.2], b: [0.3, -0.1], M: [] },
  next_sample: { x: [0.2], y: 0, city: "paris", lat: 48.8, lon: 2.3 },
  samples: [
    { x: [0.2], y: 0, city: "paris", lat: 48.8, lon: 2.3 },
    { x: [0.3], y: 1, city: "madrid", lat: 40.4, lon: -3.7 },
  ],
};

describe("TinyGpsCityMap", () => {
  it("renders map title and legend", () => {
    render(<TinyGpsCityMap state={mockState} />);

    expect(screen.getByText(/City Map \+ Boundaries/)).toBeInTheDocument();
    expect(screen.getAllByText("paris").length).toBeGreaterThan(0);
  });
});
