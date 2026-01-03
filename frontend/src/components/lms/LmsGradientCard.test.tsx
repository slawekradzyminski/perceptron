import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { LmsGradientCard } from "./LmsGradientCard";

const history = [
  {
    x: [-1, -1],
    y: -1,
    w_before: [0.2, -0.1],
    b_before: 0,
    y_hat: 0,
    error: -1,
    grad_w1: 2,
    grad_w2: -1,
    grad_b: 2,
    w_after: [0.3, -0.2],
    b_after: -0.1,
    idx: 1,
    lr: 0.1,
  },
];

test("renders gradient card with vector info", () => {
  render(<LmsGradientCard history={history} />);
  expect(screen.getByText("Gradient view")).toBeInTheDocument();
  expect(screen.getByText(/∇E/)).toBeInTheDocument();
  expect(screen.getByText(/Δw/)).toBeInTheDocument();
});
