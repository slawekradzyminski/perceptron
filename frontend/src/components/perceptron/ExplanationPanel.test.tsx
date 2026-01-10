import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { ExplanationPanel } from "./ExplanationPanel";

test("renders explanation panel", () => {
  render(<ExplanationPanel />);
  expect(screen.getByText(/input grid shows the switches/i)).toBeInTheDocument();
  expect(screen.getByText(/XOR.*no single line/i)).toBeInTheDocument();
});
