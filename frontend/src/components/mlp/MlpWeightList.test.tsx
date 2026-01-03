import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { MlpWeightList } from "./MlpWeightList";
import { mockMlpSnapshot } from "../../testutils/mocks/apiResponses";

test("renders weight values", () => {
  const snapshot = mockMlpSnapshot();
  render(<MlpWeightList snapshot={snapshot} />);
  expect(screen.getByText("Weights")).toBeInTheDocument();
  expect(screen.getByText("w₁₁¹")).toBeInTheDocument();
  expect(screen.getByText("0.32")).toBeInTheDocument();
});
