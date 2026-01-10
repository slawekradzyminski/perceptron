import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { AlexNetControls } from "./AlexNetControls";
import type { AlexNetState, FilterGrid } from "../../hooks/alexnet/useAlexNetApi";

const mockState: AlexNetState = {
  current_layer: 1,
  sample_images: ["gradient"],
  layers: [
    { layer: 1, name: "Conv1", filters: 64, kernel_size: 11, in_channels: 3 },
  ],
  param_count: 61000000,
};

const mockLayerInfo: FilterGrid["layer_info"] = {
  name: "Conv1",
  filters: 64,
  kernel_size: 11,
  in_channels: 3,
};

describe("AlexNetControls", () => {
  it("renders controls and handles interactions", () => {
    const handleLayerChange = vi.fn();
    const handleSampleChange = vi.fn();
    const handleFileUpload = vi.fn();

    render(
      <AlexNetControls
        layers={[1, 2]}
        samples={["gradient", "noise"]}
        selectedLayer={1}
        selectedSample="gradient"
        layerInfo={mockLayerInfo}
        inputImage="dGVzdA=="
        customImageUrl={null}
        state={mockState}
        loading={false}
        onLayerChange={handleLayerChange}
        onSampleChange={handleSampleChange}
        onFileUpload={handleFileUpload}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "2" }));
    expect(handleLayerChange).toHaveBeenCalledWith(2);

    fireEvent.click(screen.getByRole("button", { name: "noise" }));
    expect(handleSampleChange).toHaveBeenCalledWith("noise");

    const fileInput = screen.getByTestId("alexnet-upload-input") as HTMLInputElement;
    const file = new File(["test"], "test.png", { type: "image/png" });
    fireEvent.change(fileInput, { target: { files: [file] } });
    expect(handleFileUpload).toHaveBeenCalledWith(file);

    expect(screen.getByText("61.0M")).toBeInTheDocument();
  });
});
