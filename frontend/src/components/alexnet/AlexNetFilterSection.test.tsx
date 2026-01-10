import { render, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { AlexNetFilterSection } from "./AlexNetFilterSection";
import type { FilterGrid } from "../../hooks/alexnet/useAlexNetApi";

const mockFilters: FilterGrid = {
  rows: 1,
  cols: 2,
  count: 2,
  layer: 1,
  layer_info: { name: "Conv1", filters: 64, kernel_size: 11, in_channels: 3 },
  filters: [
    { index: 0, kernel_size: 11, image: "dGVzdA==" },
    { index: 1, kernel_size: 11, image: "dGVzdA==" },
  ],
};

describe("AlexNetFilterSection", () => {
  it("notifies hover changes", () => {
    const handleHover = vi.fn();
    const { container } = render(
      <AlexNetFilterSection
        filters={mockFilters}
        selectedLayer={1}
        hoveredFilter={null}
        onHoverChange={handleHover}
      />,
    );

    const filterCells = container.querySelectorAll(".filter-cell");
    fireEvent.mouseEnter(filterCells[0]);
    expect(handleHover).toHaveBeenCalledWith(0);

    fireEvent.mouseLeave(filterCells[0]);
    expect(handleHover).toHaveBeenCalledWith(null);
  });
});
