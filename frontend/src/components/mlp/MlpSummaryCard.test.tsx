import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { MlpSummaryCard } from "./MlpSummaryCard";
import { mockMlpSnapshot } from "../../testutils/mocks/apiResponses";

test("renders summary values", () => {
  const snapshot = mockMlpSnapshot();
  render(<MlpSummaryCard snapshot={snapshot} correct={2} total={4} />);
  expect(screen.getByText("Dataset")).toBeInTheDocument();
  expect(screen.getByText("XOR")).toBeInTheDocument();
  expect(screen.getByText("Target y")).toBeInTheDocument();
  expect(screen.getByText("-1")).toBeInTheDocument();
  expect(screen.getByText("2/4")).toBeInTheDocument();
});
