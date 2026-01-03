import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { ErrorSurfacePanel } from "./ErrorSurfacePanel";

function mockCanvas() {
  HTMLCanvasElement.prototype.getContext = () =>
    ({
      canvas: { width: 100, height: 100 },
      clearRect: () => undefined,
      fillRect: () => undefined,
      beginPath: () => undefined,
      moveTo: () => undefined,
      lineTo: () => undefined,
      stroke: () => undefined,
      set fillStyle(_: string) {},
      set strokeStyle(_: string) {},
      set lineWidth(_: number) {},
    }) as unknown as CanvasRenderingContext2D;
}

test("renders error surface panel with heatmap", () => {
  mockCanvas();
  render(
    <ErrorSurfacePanel
      data={{
        dataset: "or",
        grid_rows: 1,
        grid_cols: 2,
        steps: 3,
        w_range: [-1, 1],
        bias: 0,
        sample_count: 4,
        grid: [
          [0.1, 0.2, 0.3],
          [0.2, 0.3, 0.4],
          [0.3, 0.4, 0.5],
        ],
      }}
      loading={false}
      error={null}
      config={{ steps: 3, wMin: -1, wMax: 1, bias: 0 }}
      canShow
      canFetch
      onConfigChange={() => undefined}
    />,
  );

  expect(screen.getByText("Error Surface")).toBeInTheDocument();
  expect(screen.getByLabelText("Error surface heatmap")).toBeInTheDocument();
  expect(screen.getByDisplayValue("3")).toBeInTheDocument();
});

test("shows hover tooltip with w1/w2/loss", () => {
  mockCanvas();
  render(
    <ErrorSurfacePanel
      data={{
        dataset: "or",
        grid_rows: 1,
        grid_cols: 2,
        steps: 3,
        w_range: [-1, 1],
        bias: 0,
        sample_count: 4,
        grid: [
          [0.1, 0.2, 0.3],
          [0.2, 0.3, 0.4],
          [0.3, 0.4, 0.5],
        ],
      }}
      loading={false}
      error={null}
      config={{ steps: 3, wMin: -1, wMax: 1, bias: 0 }}
      canShow
      canFetch
      onConfigChange={() => undefined}
    />,
  );

  const canvas = screen.getByLabelText("Error surface heatmap") as HTMLCanvasElement;
  canvas.getBoundingClientRect = () =>
    ({
      left: 0,
      top: 0,
      width: 100,
      height: 100,
      right: 100,
      bottom: 100,
      x: 0,
      y: 0,
      toJSON: () => "",
    }) as DOMRect;

  fireEvent.mouseMove(canvas, { clientX: 50, clientY: 50 });
  const tooltip = screen.getByTestId("heatmap-tooltip");
  expect(tooltip.textContent).toContain("w1=");
  expect(tooltip.textContent).toContain("w2=");
  expect(tooltip.textContent).toContain("loss=");
});
