import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { DeepNetworkPreview } from "./DeepNetworkPreview";

describe("DeepNetworkPreview", () => {
  beforeEach(() => {
    const context = {
      clearRect: () => undefined,
      beginPath: () => undefined,
      arc: () => undefined,
      fill: () => undefined,
      stroke: () => undefined,
      moveTo: () => undefined,
      lineTo: () => undefined,
      fillText: () => undefined,
      strokeStyle: "",
      lineWidth: 0,
      globalAlpha: 1,
      font: "",
      textAlign: "center" as CanvasTextAlign,
      textBaseline: "alphabetic" as CanvasTextBaseline,
      fillStyle: "",
    } as unknown as CanvasRenderingContext2D;
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => context);
  });

  it("renders architecture summary and hover diagram", () => {
    render(<DeepNetworkPreview inputDim={2} hiddenDims={[4, 4]} outputDim={1} />);

    expect(screen.getByText("2 → [4 × 4] → 1")).toBeInTheDocument();

    const preview = screen.getByText(/hover for diagram/i);
    fireEvent.mouseEnter(preview.parentElement as HTMLElement);

    expect(screen.getByText("Network Architecture")).toBeInTheDocument();
  });
});
