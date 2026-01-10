import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ConvStepMath } from "./ConvStepMath";
import type { ConvStepResult } from "../../hooks/conv/useConvApi";

describe("ConvStepMath", () => {
  const mockStep: ConvStepResult = {
    output_row: 1,
    output_col: 2,
    input_row: 1,
    input_col: 2,
    patch: [
      [100, 150, 200],
      [50, 100, 150],
      [0, 50, 100],
    ],
    kernel: [
      [1, 0, -1],
      [2, 0, -2],
      [1, 0, -1],
    ],
    products: [
      [100, 0, -200],
      [100, 0, -300],
      [0, 0, -100],
    ],
    sum: -400,
    kernel_size: 3,
    padding: 1,
    stride: 1,
  };

  it("shows instruction when no step is provided", () => {
    render(<ConvStepMath step={null} />);

    expect(
      screen.getByText("Hover over activation map to see calculation")
    ).toBeInTheDocument();
  });

  it("shows step position when step is provided", () => {
    render(<ConvStepMath step={mockStep} />);

    expect(screen.getByText("Step Calculation at (1, 2)")).toBeInTheDocument();
  });

  it("renders receptive field section", () => {
    render(<ConvStepMath step={mockStep} />);

    expect(screen.getByText("Receptive Field (Patch)")).toBeInTheDocument();
  });

  it("renders kernel weights section", () => {
    render(<ConvStepMath step={mockStep} />);

    expect(screen.getByText("Kernel Weights")).toBeInTheDocument();
  });

  it("renders element-wise products section", () => {
    render(<ConvStepMath step={mockStep} />);

    expect(screen.getByText("Element-wise Products")).toBeInTheDocument();
  });

  it("displays the sum result", () => {
    render(<ConvStepMath step={mockStep} />);

    expect(screen.getByText("Sum:")).toBeInTheDocument();
    expect(screen.getByText("-400.00")).toBeInTheDocument();
  });

  it("renders math operators", () => {
    render(<ConvStepMath step={mockStep} />);

    const mathOps = document.querySelectorAll(".math-op");
    expect(mathOps.length).toBe(2);
    expect(mathOps[0]).toHaveTextContent("×");
    expect(mathOps[1]).toHaveTextContent("=");
  });

  it("displays patch values correctly", () => {
    render(<ConvStepMath step={mockStep} />);

    const patchGrid = document.querySelector(".patch-grid");
    expect(patchGrid).toBeInTheDocument();

    // Check some patch values are displayed
    expect(patchGrid?.textContent).toContain("100");
    expect(patchGrid?.textContent).toContain("150");
    expect(patchGrid?.textContent).toContain("200");
  });
});
