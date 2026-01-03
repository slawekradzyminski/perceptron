import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { StepMathPanel } from "./StepMathPanel";

const baseProps = {
  datasetName: "or",
  w: [0, 0],
  b: 0,
  sampleIdx: 0,
  nextInput: { x: [-1, -1], y: -1 },
  fallbackInput: [-1, -1],
  lastStep: null,
  apiError: null,
};

test("renders step math without last step", () => {
  render(<StepMathPanel {...baseProps} />);
  expect(screen.getByText("Step math")).toBeInTheDocument();
  expect(screen.getByText("Current state")).toBeInTheDocument();
});

test("renders last step calculation", () => {
  render(
    <StepMathPanel
      {...baseProps}
      lastStep={{
        x: [-1, 1],
        y: 1,
        score: 1,
        pred: 1,
        mistake: false,
        deltaW: [0, 0],
        deltaB: 0,
        lr: 1,
      }}
    />,
  );
  expect(screen.getByText("Last step calculation")).toBeInTheDocument();
});
