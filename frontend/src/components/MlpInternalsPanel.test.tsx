import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { MlpInternalsPanel } from "./MlpInternalsPanel";

function mockCanvas() {
  HTMLCanvasElement.prototype.getContext = () =>
    ({
      canvas: { width: 100, height: 100 },
      clearRect: () => undefined,
      fillRect: () => undefined,
      beginPath: () => undefined,
      moveTo: () => undefined,
      lineTo: () => undefined,
      stroke: () => undefined,
      set fillStyle(_: string) {},
      set strokeStyle(_: string) {},
      set lineWidth(_: number) {},
    }) as unknown as CanvasRenderingContext2D;
}

test("renders mlp internals panel", () => {
  mockCanvas();
  render(
    <MlpInternalsPanel
      data={{
        dataset: "xor",
        grid_rows: 1,
        grid_cols: 2,
        hidden_dim: 2,
        sample_index: 0,
        sample_count: 4,
        x: [1, -1],
        y: 1,
        y01: 1,
        loss: 0.2,
        p_hat: 0.6,
        hidden: {
          weights_before: [[0.1, 0.2], [0.3, 0.4]],
          bias_before: [0, 0],
          weights_after: [[0.2, 0.3], [0.4, 0.5]],
          bias_after: [0.1, 0.1],
          z: [0.1, 0.2],
          a: [0.1, 0.2],
          templates_before: [[[0.1, 0.2]], [[0.3, 0.4]]],
          templates_after: [[[0.2, 0.3]], [[0.4, 0.5]]],
        },
        output: {
          weights_before: [[0.2, 0.3]],
          bias_before: [0],
          weights_after: [[0.3, 0.4]],
          bias_after: [0.1],
          z: 0.2,
          a: 0.6,
        },
        gradients: {
          hidden_W: [[0.1, 0.1], [0.2, 0.2]],
          hidden_b: [0.1, 0.1],
          out_W: [[0.1, 0.1]],
          out_b: [0.1],
          templates: [[[0.1, 0.1]], [[0.2, 0.2]]],
        },
      }}
      loading={false}
      error={null}
      config={{ hiddenDim: 2, sampleIndex: 0, lr: 0.5, seed: 0 }}
      canFetch
      onConfigChange={() => undefined}
    />,
  );

  expect(screen.getByText("MLP Internals")).toBeInTheDocument();
  expect(screen.getByText("What the model does")).toBeInTheDocument();
  expect(screen.getByText("Hidden weights (before)")).toBeInTheDocument();
  expect(screen.getByText("Hidden gradients")).toBeInTheDocument();
});
