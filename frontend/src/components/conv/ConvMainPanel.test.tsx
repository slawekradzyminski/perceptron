import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ConvMainPanel } from "./ConvMainPanel";
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

describe("ConvMainPanel", () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({ ok: true }) as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders main convolution panels", async () => {
    render(
      <ConvMainPanel
        state={mockState}
        step={null}
        apiBase="http://localhost:8000"
        hoveredCell={null}
        receptiveField={null}
        onCellHover={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("🔢 Digit Recognition")).toBeInTheDocument();
    });
    expect(
      screen.getByText("How Convolutional Neural Networks Work"),
    ).toBeInTheDocument();
  });
});
