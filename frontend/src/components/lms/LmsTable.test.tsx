import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { LmsTable } from "./LmsTable";

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

test("renders LMS table and handles hover", () => {
  const onHover = vi.fn();
  const onLeave = vi.fn();
  render(
    <LmsTable
      history={history}
      stepCount={1}
      firstStepNumber={1}
      onHover={onHover}
      onLeave={onLeave}
    />,
  );
  expect(screen.getByText("Step")).toBeInTheDocument();
  const row = screen.getAllByRole("row")[1];
  fireEvent.mouseMove(row, { clientX: 50, clientY: 50 });
  expect(onHover).toHaveBeenCalled();
  fireEvent.mouseLeave(row);
  expect(onLeave).toHaveBeenCalled();
});
