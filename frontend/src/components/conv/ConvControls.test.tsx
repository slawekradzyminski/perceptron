import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ConvControls } from "./ConvControls";
import type { ConvState } from "../../hooks/conv/useConvApi";

const mockState: ConvState = {
  image: [
    [0, 255],
    [255, 0],
  ],
  image_name: "checkerboard",
  image_size: 8,
  kernel: [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ],
  kernel_name: "identity",
  padding: 1,
  stride: 1,
  input_shape: [8, 8],
  kernel_shape: [3, 3],
  output_shape: [8, 8],
  activation_map: [[0]],
  activation_map_normalized: [[0]],
  image_base64: "",
  activation_base64: "",
};

describe("ConvControls", () => {
  const defaultProps = {
    state: mockState,
    loading: false,
    currentKernel: mockState.kernel,
    editingKernel: null,
    onImageChange: vi.fn(),
    onImageSizeChange: vi.fn(),
    onFileUpload: vi.fn(),
    onKernelChange: vi.fn(),
    onKernelWeightChange: vi.fn(),
    onApplyCustomKernel: vi.fn(),
    onPaddingChange: vi.fn(),
    onStrideChange: vi.fn(),
  };

  it("renders sample image buttons", () => {
    render(<ConvControls {...defaultProps} />);

    expect(screen.getByText("gradient")).toBeInTheDocument();
    expect(screen.getByText("checkerboard")).toBeInTheDocument();
    expect(screen.getByText("cross")).toBeInTheDocument();
  });

  it("renders kernel preset buttons", () => {
    render(<ConvControls {...defaultProps} />);

    expect(screen.getByText("identity")).toBeInTheDocument();
    expect(screen.getByText("edge horizontal")).toBeInTheDocument();
    expect(screen.getByText("sobel x")).toBeInTheDocument();
  });

  it("calls onImageChange when clicking a sample image", () => {
    render(<ConvControls {...defaultProps} />);

    fireEvent.click(screen.getByText("gradient"));
    expect(defaultProps.onImageChange).toHaveBeenCalledWith("gradient");
  });

  it("calls onKernelChange when clicking a kernel preset", () => {
    render(<ConvControls {...defaultProps} />);

    fireEvent.click(screen.getByText("blur"));
    expect(defaultProps.onKernelChange).toHaveBeenCalledWith("blur");
  });

  it("calls onPaddingChange when clicking padding button", () => {
    render(<ConvControls {...defaultProps} />);

    // Click padding 0 button
    const paddingButtons = screen.getAllByRole("button").filter(
      (btn) => btn.textContent === "0"
    );
    fireEvent.click(paddingButtons[0]);
    expect(defaultProps.onPaddingChange).toHaveBeenCalledWith(0);
  });

  it("calls onStrideChange when clicking stride button", () => {
    const props = {
      ...defaultProps,
      onStrideChange: vi.fn(),
    };
    render(<ConvControls {...props} />);

    // Click stride 3 button (which is always available in the UI)
    const strideButtons = screen.getAllByRole("button").filter(
      (btn) => btn.textContent === "3"
    );
    fireEvent.click(strideButtons[0]);
    expect(props.onStrideChange).toHaveBeenCalledWith(3);
  });

  it("displays current dimensions", () => {
    render(<ConvControls {...defaultProps} />);

    expect(screen.getByText("Current Dimensions")).toBeInTheDocument();
    // There are multiple 8×8 elements (input and output), check at least one exists
    expect(screen.getAllByText("8×8").length).toBeGreaterThan(0);
    expect(screen.getByText("3×3")).toBeInTheDocument();
  });

  it("shows Apply button when editing kernel", () => {
    render(
      <ConvControls
        {...defaultProps}
        editingKernel={[
          [1, 0, -1],
          [0, 0, 0],
          [-1, 0, 1],
        ]}
      />
    );

    expect(screen.getByText("Apply Custom Kernel")).toBeInTheDocument();
  });

  it("hides Apply button when not editing", () => {
    render(<ConvControls {...defaultProps} />);

    expect(screen.queryByText("Apply Custom Kernel")).not.toBeInTheDocument();
  });

  it("disables buttons when loading", () => {
    render(<ConvControls {...defaultProps} loading={true} />);

    const gradientButton = screen.getByText("gradient");
    expect(gradientButton).toBeDisabled();
  });

  it("renders kernel editor inputs", () => {
    render(<ConvControls {...defaultProps} />);

    const kernelInputs = screen.getAllByRole("spinbutton");
    expect(kernelInputs.length).toBe(9); // 3x3 kernel
  });

  it("calls onKernelWeightChange when editing kernel value", () => {
    render(<ConvControls {...defaultProps} />);

    const kernelInputs = screen.getAllByRole("spinbutton");
    fireEvent.change(kernelInputs[0], { target: { value: "0.5" } });

    expect(defaultProps.onKernelWeightChange).toHaveBeenCalledWith(0, 0, "0.5");
  });
});
