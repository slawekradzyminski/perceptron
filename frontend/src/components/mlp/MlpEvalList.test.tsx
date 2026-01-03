import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { MlpEvalList } from "./MlpEvalList";
import { mockMlpSnapshot } from "../../testutils/mocks/apiResponses";

test("renders evaluation rows", () => {
  const snapshot = mockMlpSnapshot();
  render(<MlpEvalList evals={snapshot.evals} />);
  expect(screen.getByText("Predictions on dataset")).toBeInTheDocument();
  expect(screen.getByText(/p̂=0.480/)).toBeInTheDocument();
});
