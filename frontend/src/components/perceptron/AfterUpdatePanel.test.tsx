import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { AfterUpdatePanel } from "./AfterUpdatePanel";

function mockCanvas() {
  HTMLCanvasElement.prototype.getContext = () =>
    ({
      canvas: { width: 100, height: 50 },
      clearRect: () => undefined,
      fillRect: () => undefined,
      beginPath: () => undefined,
      moveTo: () => undefined,
      lineTo: () => undefined,
      stroke: () => undefined,
      setLineDash: () => undefined,
      set fillStyle(_: string) {},
      set strokeStyle(_: string) {},
      set lineWidth(_: number) {},
    }) as unknown as CanvasRenderingContext2D;
}

test("renders bias after and triggers tooltip", () => {
  mockCanvas();
  const weightsAfterRef = { current: document.createElement("canvas") } as React.RefObject<HTMLCanvasElement>;
  const onTooltipChange = vi.fn();
  const onTooltipHide = vi.fn();

  render(
    <AfterUpdatePanel
      gridRows={1}
      gridCols={2}
      weightsAfter={[[1, -1]]}
      biasAfter={1}
      lastDeltaW={[0, 0]}
      weightsAfterRef={weightsAfterRef}
      onTooltipChange={onTooltipChange}
      onTooltipHide={onTooltipHide}
    />,
  );

  expect(screen.getByText("Bias (after)")).toBeInTheDocument();
  const canvas = screen.getByText("Weights (after)").parentElement?.querySelector("canvas");
  if (!canvas) throw new Error("Canvas missing");
  Object.defineProperty(canvas, "getBoundingClientRect", {
    value: () => ({ left: 0, top: 0, width: 100, height: 50 }),
  });
  fireEvent.mouseMove(canvas, { clientX: 10, clientY: 10 });
  expect(onTooltipChange).toHaveBeenCalled();
});
