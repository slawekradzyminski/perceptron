import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { MlpStepTemplates } from "./MlpStepTemplates";
import { mockMlpInternals } from "../../testutils/mocks/apiResponses";

test("renders step templates", () => {
  render(<MlpStepTemplates step={mockMlpInternals()} />);
  expect(screen.getByText("Last step templates")).toBeInTheDocument();
  expect(screen.getByText("H1 (before)")).toBeInTheDocument();
});
