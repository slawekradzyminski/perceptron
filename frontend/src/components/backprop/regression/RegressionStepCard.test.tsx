import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RegressionStepCard } from "./RegressionStepCard";
import type { RegressionState, RegressionStep } from "../../../types";

const mockState: RegressionState = {
  loss: "mse",
  idx: 0,
  lr: 0.1,
  sample_count: 1,
  params: { m: 1, b: 1 },
  next_sample: { x: 1, y: 2 },
  samples: [{ x: 1, y: 2 }],
  sample_order: [0],
};

const mockStep: RegressionStep = {
  loss: "mse",
  idx: 0,
  sample_idx: 0,
  lr: 0.1,
  sample: { x: 1, y: 2 },
  y_hat: 2,
  loss_value: 1,
  grads: { m: 0.5, b: 0.1 },
  params_before: { m: 1, b: 1 },
  params_after: { m: 1.1, b: 1.2 },
  mean_loss: 1,
  sample_count: 1,
};

describe("RegressionStepCard", () => {
  it("renders placeholder without sample", () => {
    render(<RegressionStepCard state={null} step={null} />);
    expect(screen.getByText(/Step regression/)).toBeInTheDocument();
  });

  it("renders step data", () => {
    render(<RegressionStepCard state={mockState} step={mockStep} />);
    expect(screen.getByText("Regression Step")).toBeInTheDocument();
    expect(screen.getByText(/x=1.000/)).toBeInTheDocument();
  });
});
