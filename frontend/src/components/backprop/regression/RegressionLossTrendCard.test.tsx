import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RegressionLossTrendCard } from "./RegressionLossTrendCard";
import type { RegressionStep } from "../../../types";

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

describe("RegressionLossTrendCard", () => {
  it("renders placeholder when empty", () => {
    render(<RegressionLossTrendCard history={[]} lossType="mse" />);
    expect(screen.getByText(/Step through regression/)).toBeInTheDocument();
  });

  it("renders loss chart", () => {
    render(<RegressionLossTrendCard history={[mockStep]} lossType="mse" />);
    expect(screen.getByLabelText("Loss trend")).toBeInTheDocument();
  });
});
