import { render, screen } from "@testing-library/react";
import { test, expect } from "vitest";
import { LmsMathCard } from "./LmsMathCard";

test("renders LMS math card", () => {
  render(<LmsMathCard />);
  expect(screen.getByText("Update rule")).toBeInTheDocument();
  expect(screen.getByText(/Prediction:/)).toBeInTheDocument();
  expect(screen.getByText(/Gradient:/)).toBeInTheDocument();
});
