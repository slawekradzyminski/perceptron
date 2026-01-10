import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { TemplateGrid } from "./TemplateGrid";
import * as visuals from "../../utils/visuals";

vi.mock("../../utils/visuals", () => ({
  drawGrid: vi.fn(),
  setGridCanvasSize: vi.fn(),
  valueColor: vi.fn(),
}));

describe("TemplateGrid", () => {
  beforeEach(() => {
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      fillRect: vi.fn(),
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  it("renders label and draws grid", () => {
    render(<TemplateGrid label="Template" grid={[[1, -1], [0, 1]]} />);

    expect(screen.getByText("Template")).toBeInTheDocument();
    expect(visuals.setGridCanvasSize).toHaveBeenCalled();
    expect(visuals.drawGrid).toHaveBeenCalled();
  });
});
