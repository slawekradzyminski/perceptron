import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RegressionBookTable } from "./RegressionBookTable";
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

describe("RegressionBookTable", () => {
  it("renders placeholder when empty", () => {
    render(
      <RegressionBookTable history={[]} lossType="mse" bookLiteral={false} />,
    );
    expect(screen.getByText(/Run steps to fill the table/)).toBeInTheDocument();
  });

  it("renders table and hover details", () => {
    render(
      <RegressionBookTable
        history={[mockStep]}
        lossType="mse"
        bookLiteral={false}
      />,
    );

    fireEvent.mouseEnter(screen.getByText("0"));
    expect(screen.getByText(/Step index/)).toBeInTheDocument();
  });
});
