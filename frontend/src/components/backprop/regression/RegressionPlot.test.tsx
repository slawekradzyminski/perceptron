import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RegressionPlot } from "./RegressionPlot";
import type { RegressionState, RegressionStep } from "../../../types";

const mockState: RegressionState = {
  loss: "mse",
  idx: 0,
  lr: 0.1,
  sample_count: 2,
  params: { m: 1, b: 1 },
  next_sample: { x: 1, y: 2 },
  samples: [
    { x: 1, y: 2 },
    { x: 2, y: 5 },
  ],
  sample_order: [0, 1],
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
  sample_count: 2,
};

describe("RegressionPlot", () => {
  it("renders placeholder when no samples", () => {
    render(<RegressionPlot state={{ ...mockState, samples: [] }} history={[]} />);
    expect(screen.getByText(/Regression samples/)).toBeInTheDocument();
  });

  it("renders plot with samples", () => {
    render(<RegressionPlot state={mockState} history={[mockStep]} />);
    expect(screen.getByText(/Fitted Line/)).toBeInTheDocument();
    expect(screen.getByText(/Current/)).toBeInTheDocument();
  });
});
