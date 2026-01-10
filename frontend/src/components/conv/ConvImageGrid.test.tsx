import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ConvImageGrid } from "./ConvImageGrid";

describe("ConvImageGrid", () => {
  const defaultProps = {
    image: [
      [0, 128, 255],
      [64, 192, 32],
      [255, 0, 128],
    ],
    imageSize: 3,
    imageName: "test_image",
    inputShape: [3, 3] as [number, number],
    receptiveField: null,
  };

  it("renders input image section", () => {
    render(<ConvImageGrid {...defaultProps} />);

    expect(screen.getByText("Input Image")).toBeInTheDocument();
  });

  it("renders all pixel cells", () => {
    render(<ConvImageGrid {...defaultProps} />);

    // 3x3 = 9 cells
    const cells = document.querySelectorAll(".pixel-cell");
    expect(cells.length).toBe(9);
  });

  it("displays image name and dimensions", () => {
    render(<ConvImageGrid {...defaultProps} />);

    expect(screen.getByText("test_image (3×3)")).toBeInTheDocument();
  });

  it("highlights cells within receptive field", () => {
    render(
      <ConvImageGrid
        {...defaultProps}
        receptiveField={{
          startRow: 0,
          startCol: 0,
          endRow: 2,
          endCol: 2,
        }}
      />
    );

    const highlightedCells = document.querySelectorAll(".pixel-cell.highlighted");
    expect(highlightedCells.length).toBe(4); // 2x2 region
  });

  it("does not highlight cells outside receptive field", () => {
    render(
      <ConvImageGrid
        {...defaultProps}
        receptiveField={{
          startRow: 1,
          startCol: 1,
          endRow: 3,
          endCol: 3,
        }}
      />
    );

    const highlightedCells = document.querySelectorAll(".pixel-cell.highlighted");
    expect(highlightedCells.length).toBe(4); // Only 2x2 bottom-right
  });

  it("sets correct background color based on pixel value", () => {
    render(<ConvImageGrid {...defaultProps} />);

    const cells = document.querySelectorAll(".pixel-cell");
    // First cell has value 0 (black)
    expect(cells[0]).toHaveStyle("background-color: rgb(0, 0, 0)");
    // Third cell has value 255 (white)
    expect(cells[2]).toHaveStyle("background-color: rgb(255, 255, 255)");
  });
});
