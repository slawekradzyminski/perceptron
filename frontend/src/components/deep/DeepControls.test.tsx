import { render, screen, fireEvent } from "@testing-library/react";
import { expect, test, vi, describe } from "vitest";
import { DeepControls } from "./DeepControls";

const defaultProps = {
  dataset: "circles",
  datasets: ["circles", "spiral", "xor", "baarle"],
  hiddenDims: [8, 8],
  lr: 0.1,
  seed: 42,
  loading: false,
  onDatasetChange: vi.fn(),
  onHiddenDimsChange: vi.fn(),
  onLrChange: vi.fn(),
  onSeedChange: vi.fn(),
};

describe("DeepControls", () => {
  test("renders architecture header", () => {
    render(<DeepControls {...defaultProps} />);
    expect(screen.getByText("Architecture")).toBeInTheDocument();
  });

  test("renders dataset selector with options", () => {
    render(<DeepControls {...defaultProps} />);
    expect(screen.getByText("Dataset")).toBeInTheDocument();
    const select = screen.getByRole("combobox");
    expect(select).toBeInTheDocument();
    expect(select).toHaveValue("circles");
  });

  test("renders all dataset options", () => {
    render(<DeepControls {...defaultProps} />);
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(4);
    expect(options[0]).toHaveValue("circles");
    expect(options[1]).toHaveValue("spiral");
    expect(options[2]).toHaveValue("xor");
    expect(options[3]).toHaveValue("baarle");
  });

  test("calls onDatasetChange when dataset is changed", () => {
    const onDatasetChange = vi.fn();
    render(<DeepControls {...defaultProps} onDatasetChange={onDatasetChange} />);

    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "xor" } });

    expect(onDatasetChange).toHaveBeenCalledWith("xor");
  });

  test("renders depth slider with current value", () => {
    render(<DeepControls {...defaultProps} />);
    expect(screen.getByText("Depth (layers)")).toBeInTheDocument();
    const depthSlider = screen.getAllByRole("slider")[0];
    expect(depthSlider).toHaveValue("2"); // hiddenDims.length
  });

  test("calls onHiddenDimsChange when depth changes", () => {
    const onHiddenDimsChange = vi.fn();
    render(<DeepControls {...defaultProps} onHiddenDimsChange={onHiddenDimsChange} />);

    const depthSlider = screen.getAllByRole("slider")[0];
    fireEvent.change(depthSlider, { target: { value: "3" } });

    // Should create array with 3 elements of width 8
    expect(onHiddenDimsChange).toHaveBeenCalledWith([8, 8, 8]);
  });

  test("renders width slider with current value", () => {
    render(<DeepControls {...defaultProps} />);
    expect(screen.getByText("Width (neurons)")).toBeInTheDocument();
    const widthSlider = screen.getAllByRole("slider")[1];
    expect(widthSlider).toHaveValue("8");
  });

  test("calls onHiddenDimsChange when width changes", () => {
    const onHiddenDimsChange = vi.fn();
    render(<DeepControls {...defaultProps} onHiddenDimsChange={onHiddenDimsChange} />);

    const widthSlider = screen.getAllByRole("slider")[1];
    fireEvent.change(widthSlider, { target: { value: "16" } });

    // Should create array with 2 elements of width 16
    expect(onHiddenDimsChange).toHaveBeenCalledWith([16, 16]);
  });

  test("renders learning rate input", () => {
    render(<DeepControls {...defaultProps} />);
    expect(screen.getByText("Learning Rate")).toBeInTheDocument();
    const lrInput = screen.getByDisplayValue("0.1");
    expect(lrInput).toBeInTheDocument();
  });

  test("calls onLrChange when learning rate changes", () => {
    const onLrChange = vi.fn();
    render(<DeepControls {...defaultProps} onLrChange={onLrChange} />);

    const lrInput = screen.getByDisplayValue("0.1");
    fireEvent.change(lrInput, { target: { value: "0.5" } });

    expect(onLrChange).toHaveBeenCalledWith(0.5);
  });

  test("renders seed input", () => {
    render(<DeepControls {...defaultProps} />);
    expect(screen.getByText("Seed")).toBeInTheDocument();
    const seedInput = screen.getByDisplayValue("42");
    expect(seedInput).toBeInTheDocument();
  });

  test("calls onSeedChange when seed changes", () => {
    const onSeedChange = vi.fn();
    render(<DeepControls {...defaultProps} onSeedChange={onSeedChange} />);

    const seedInput = screen.getByDisplayValue("42");
    fireEvent.change(seedInput, { target: { value: "123" } });

    expect(onSeedChange).toHaveBeenCalledWith(123);
  });

  test("renders architecture preview", () => {
    render(<DeepControls {...defaultProps} />);
    expect(screen.getByText("2 → [8 × 8] → 2")).toBeInTheDocument();
  });

  test("disables controls when loading", () => {
    render(<DeepControls {...defaultProps} loading={true} />);

    const select = screen.getByRole("combobox");
    expect(select).toBeDisabled();

    const sliders = screen.getAllByRole("slider");
    sliders.forEach((slider) => {
      expect(slider).toBeDisabled();
    });
  });

  test("handles single layer architecture", () => {
    render(<DeepControls {...defaultProps} hiddenDims={[16]} />);
    const depthSlider = screen.getAllByRole("slider")[0];
    expect(depthSlider).toHaveValue("1");
    expect(screen.getByText("2 → [16] → 2")).toBeInTheDocument();
  });

  test("handles deep architecture", () => {
    render(<DeepControls {...defaultProps} hiddenDims={[32, 32, 32, 32]} />);
    const depthSlider = screen.getAllByRole("slider")[0];
    expect(depthSlider).toHaveValue("4");
    expect(screen.getByText("2 → [32 × 32 × 32 × 32] → 2")).toBeInTheDocument();
  });
});

