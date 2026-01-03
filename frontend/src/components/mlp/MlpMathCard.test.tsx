import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { MlpMathCard } from "./MlpMathCard";

test("renders MLP math summary", () => {
  render(<MlpMathCard />);
  expect(screen.getByText("MLP (2-layer) training")).toBeInTheDocument();
  expect(screen.getByText(/Hidden:/)).toBeInTheDocument();
  expect(screen.getByText(/Output:/)).toBeInTheDocument();
});
