import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ConvVisualizationPanel } from "./ConvVisualizationPanel";
import type { ConvState } from "../../hooks/conv/useConvApi";

const mockState: ConvState = {
  image: [
    [0, 255],
    [255, 0],
  ],
  image_name: "checkerboard",
  image_size: 2,
  kernel: [[1]],
  kernel_name: "identity",
  padding: 0,
  stride: 1,
  input_shape: [2, 2],
  kernel_shape: [1, 1],
  output_shape: [2, 2],
  activation_map: [
    [0, 255],
    [255, 0],
  ],
  activation_map_normalized: [
    [0, 255],
    [255, 0],
  ],
  image_base64: "",
  activation_base64: "",
};

describe("ConvVisualizationPanel", () => {
  it("renders input and activation sections", () => {
    render(
      <ConvVisualizationPanel
        state={mockState}
        hoveredCell={null}
        receptiveField={null}
        onCellHover={vi.fn()}
      />,
    );

    expect(screen.getByText("Input Image")).toBeInTheDocument();
    expect(screen.getByText("Activation Map")).toBeInTheDocument();
  });
});
