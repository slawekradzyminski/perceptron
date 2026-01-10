import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ConvActivationGrid } from "./ConvActivationGrid";

describe("ConvActivationGrid", () => {
  const defaultProps = {
    activationMap: [
      [0.5, 1.0],
      [0.2, 0.8],
    ],
    activationMapNormalized: [
      [128, 255],
      [51, 204],
    ],
    outputShape: [2, 2] as [number, number],
    hoveredCell: null,
    onCellHover: vi.fn(),
  };

  it("renders activation map section", () => {
    render(<ConvActivationGrid {...defaultProps} />);

    expect(screen.getByText("Activation Map")).toBeInTheDocument();
  });

  it("renders all activation cells", () => {
    render(<ConvActivationGrid {...defaultProps} />);

    // 2x2 = 4 cells
    const cells = document.querySelectorAll(".activation-cell");
    expect(cells.length).toBe(4);
  });

  it("displays output dimensions", () => {
    render(<ConvActivationGrid {...defaultProps} />);

    expect(screen.getByText("output (2×2)")).toBeInTheDocument();
  });

  it("displays hover hint", () => {
    render(<ConvActivationGrid {...defaultProps} />);

    expect(screen.getByText("Hover over cells to see calculation")).toBeInTheDocument();
  });

  it("calls onCellHover on mouseenter", () => {
    render(<ConvActivationGrid {...defaultProps} />);

    const cells = document.querySelectorAll(".activation-cell");
    fireEvent.mouseEnter(cells[0]);

    expect(defaultProps.onCellHover).toHaveBeenCalledWith({ row: 0, col: 0 });
  });

  it("calls onCellHover with null on mouseleave", () => {
    render(<ConvActivationGrid {...defaultProps} />);

    const cells = document.querySelectorAll(".activation-cell");
    fireEvent.mouseLeave(cells[0]);

    expect(defaultProps.onCellHover).toHaveBeenCalledWith(null);
  });

  it("highlights hovered cell", () => {
    render(
      <ConvActivationGrid
        {...defaultProps}
        hoveredCell={{ row: 1, col: 0 }}
      />
    );

    const cells = document.querySelectorAll(".activation-cell");
    // Cell at row 1, col 0 is index 2
    expect(cells[2]).toHaveClass("hovered");
  });

  it("returns null when output shape is zero", () => {
    const { container } = render(
      <ConvActivationGrid
        {...defaultProps}
        outputShape={[0, 0]}
      />
    );

    expect(container.querySelector(".viz-panel")).not.toBeInTheDocument();
  });
});
