import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { MlpNetworkDiagram } from "./MlpNetworkDiagram";
import { mockMlpSnapshot } from "../../testutils/mocks/apiResponses";

test("renders network diagram", () => {
  render(<MlpNetworkDiagram snapshot={mockMlpSnapshot()} />);
  expect(screen.getByRole("img", { name: "MLP diagram" })).toBeInTheDocument();
});
