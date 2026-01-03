import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { LmsTrendCard } from "./LmsTrendCard";

const history = [
  {
    x: [-1, -1],
    y: -1,
    w_before: [0, 0],
    b_before: 0,
    y_hat: 0,
    error: -1,
    grad_w1: 2,
    grad_w2: 2,
    grad_b: 2,
    w_after: [0.1, 0.1],
    b_after: -0.1,
    idx: 1,
    lr: 0.1,
  },
];

test("renders trend card", () => {
  render(<LmsTrendCard history={history} stepCount={1} />);
  expect(screen.getByText(/Error trend/)).toBeInTheDocument();
  expect(screen.getByText(/Latest step/)).toBeInTheDocument();
});
