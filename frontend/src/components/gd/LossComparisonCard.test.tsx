import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { LossComparisonCard } from "./LossComparisonCard";

test("renders loss comparison card", () => {
  const points = [
    { p: 0.1, l1: 0.9, ce: 2.3026 },
    { p: 0.9, l1: 0.1, ce: 0.1053 },
  ];
  render(<LossComparisonCard points={points} />);
  expect(screen.getByText("Loss penalty vs p(correct)")).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "Loss curves" })).toBeInTheDocument();
  expect(screen.getByText("L1 loss")).toBeInTheDocument();
  expect(screen.getByText("Cross-entropy")).toBeInTheDocument();
});
