import { render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { BoundaryPanel } from "./BoundaryPanel";

vi.mock("../../utils/visuals", () => ({
  drawBoundary: vi.fn(),
  drawPoints: vi.fn(),
}));

test("renders boundary canvas when enabled", () => {
  const canvasRef = { current: document.createElement("canvas") } as React.RefObject<HTMLCanvasElement>;
  render(
    <BoundaryPanel
      w={[0, 0]}
      b={0}
      points={[]}
      show
      canvasRef={canvasRef}
    />,
  );
  expect(screen.getByText("2D Boundary")).toBeInTheDocument();
});

test("does not render when disabled", () => {
  const canvasRef = { current: document.createElement("canvas") } as React.RefObject<HTMLCanvasElement>;
  render(
    <BoundaryPanel
      w={[0, 0]}
      b={0}
      points={[]}
      show={false}
      canvasRef={canvasRef}
    />,
  );
  expect(screen.queryByText("2D Boundary")).toBeNull();
});
