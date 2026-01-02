import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { SwitchboardPanel } from "../components/SwitchboardPanel";

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

test("renders score and shows tooltip on hover", () => {
  mockCanvas();
  const inputRef = { current: document.createElement("canvas") } as React.RefObject<HTMLCanvasElement>;
  const weightsBeforeRef = { current: document.createElement("canvas") } as React.RefObject<HTMLCanvasElement>;
  const contribRef = { current: document.createElement("canvas") } as React.RefObject<HTMLCanvasElement>;
  const onTooltipChange = vi.fn();
  const onTooltipHide = vi.fn();

  render(
    <SwitchboardPanel
      gridRows={1}
      gridCols={2}
      inputGrid={[[-1, 1]]}
      weightsBefore={[[0, 0]]}
      contribGrid={[[0, 0]]}
      biasBefore={0}
      displayScore={0}
      inputRef={inputRef}
      weightsBeforeRef={weightsBeforeRef}
      contribRef={contribRef}
      tooltip={{ visible: false, text: "", x: 0, y: 0 }}
      onTooltipChange={onTooltipChange}
      onTooltipHide={onTooltipHide}
    />,
  );

  expect(screen.getByTestId("score-value")).toHaveTextContent("0.00");
  const inputBlock = screen.getByText("Input").parentElement;
  const canvas = inputBlock?.querySelector("canvas");
  if (!canvas) throw new Error("Canvas missing");
  Object.defineProperty(canvas, "getBoundingClientRect", {
    value: () => ({ left: 0, top: 0, width: 100, height: 50 }),
  });
  fireEvent.mouseMove(canvas, { clientX: 10, clientY: 10 });
  expect(onTooltipChange).toHaveBeenCalled();
});
