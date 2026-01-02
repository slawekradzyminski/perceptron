import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { ExplanationPanel } from "../components/ExplanationPanel";

test("renders explanation panel", () => {
  render(<ExplanationPanel />);
  expect(screen.getByText("What happens on each step")).toBeInTheDocument();
  expect(screen.getByText(/Score:/)).toBeInTheDocument();
});
