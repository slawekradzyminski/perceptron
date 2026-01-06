import { render, screen } from "@testing-library/react";
import { expect, test, describe, vi, beforeEach } from "vitest";
import { DeepBoundaryCanvas } from "./DeepBoundaryCanvas";

const mockBoundary = {
  resolution: 10,
  predictions: Array(10)
    .fill(null)
    .map(() => Array(10).fill(0)),
  region_ids: Array(10)
    .fill(null)
    .map((_, j) =>
      Array(10)
        .fill(null)
        .map((_, i) => (i + j) % 5),
    ),
  region_count: 5,
  theoretical_max: 100,
};

const mockSamples = [
  { x: [0.5, 0.5], y: 0 },
  { x: [-0.5, -0.5], y: 1 },
  { x: [0.2, -0.3], y: 0 },
  { x: [-0.2, 0.3], y: 1 },
];

describe("DeepBoundaryCanvas", () => {
  beforeEach(() => {
    // Mock canvas context with all required methods
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      fillStyle: "",
      strokeStyle: "",
      lineWidth: 0,
      globalAlpha: 1,
      fillRect: vi.fn(),
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      closePath: vi.fn(),
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  test("renders canvas element", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={false}
      />,
    );

    const canvas = document.querySelector("canvas");
    expect(canvas).toBeInTheDocument();
  });

  test("canvas has correct dimensions", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={false}
      />,
    );

    const canvas = document.querySelector("canvas");
    expect(canvas).toHaveAttribute("width", "400");
    expect(canvas).toHaveAttribute("height", "400");
  });

  test("renders legend", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={false}
      />,
    );

    expect(screen.getByText("Class 0")).toBeInTheDocument();
    expect(screen.getByText("Class 1")).toBeInTheDocument();
  });

  test("shows tiling note when showTiling is true", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={true}
      />,
    );

    expect(screen.getByText("Colors = Linear Regions")).toBeInTheDocument();
  });

  test("does not show tiling note when showTiling is false", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={false}
      />,
    );

    expect(screen.queryByText("Colors = Linear Regions")).not.toBeInTheDocument();
  });

  test("renders with null boundary", () => {
    render(
      <DeepBoundaryCanvas
        boundary={null}
        samples={mockSamples}
        showTiling={false}
      />,
    );

    const canvas = document.querySelector("canvas");
    expect(canvas).toBeInTheDocument();
    // Canvas should still render but won't draw anything
  });

  test("renders with empty samples", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={[]}
        showTiling={false}
      />,
    );

    const canvas = document.querySelector("canvas");
    expect(canvas).toBeInTheDocument();
  });

  test("has boundary-canvas class", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={false}
      />,
    );

    const canvas = document.querySelector("canvas");
    expect(canvas).toHaveClass("boundary-canvas");
  });

  test("renders legend items with correct classes", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={false}
      />,
    );

    const legendDots = document.querySelectorAll(".legend-dot");
    expect(legendDots).toHaveLength(2);
    expect(legendDots[0]).toHaveClass("class-0");
    expect(legendDots[1]).toHaveClass("class-1");
  });

  test("renders in deep-boundary container", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={false}
      />,
    );

    const container = document.querySelector(".deep-boundary");
    expect(container).toBeInTheDocument();
  });

  test("renders Baarle legend when dataset is baarle", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={false}
        dataset="baarle"
      />,
    );

    expect(screen.getByText("Belgium (Baarle-Hertog)")).toBeInTheDocument();
    expect(screen.getByText("Netherlands (Baarle-Nassau)")).toBeInTheDocument();
  });

  test("renders Baarle legend dots with correct classes", () => {
    render(
      <DeepBoundaryCanvas
        boundary={mockBoundary}
        samples={mockSamples}
        showTiling={false}
        dataset="baarle"
      />,
    );

    const legendDots = document.querySelectorAll(".legend-dot");
    expect(legendDots).toHaveLength(2);
    expect(legendDots[0]).toHaveClass("belgium");
    expect(legendDots[1]).toHaveClass("netherlands");
  });
});
