import { render, screen, waitFor } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { MlpTrainerPage } from "./MlpTrainerPage";

function mockCanvas() {
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    canvas: { width: 100, height: 100 },
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    strokeRect: vi.fn(),
    stroke: vi.fn(),
    setLineDash: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}

test("renders MLP trainer page", async () => {
  mockCanvas();
  const fetchMock = vi.fn(async (url: RequestInfo) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.endsWith("/mlp/reset")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "xor",
          grid_rows: 1,
          grid_cols: 2,
          hidden_dim: 2,
          lr: 0.5,
          seed: 0,
          idx: 0,
          sample_count: 4,
          next_x: [-1, -1],
          next_y: -1,
          hidden: { weights: [[0, 0], [0, 0]], bias: [0, 0], templates: [[[0, 0]], [[0, 0]]] },
          output: { weights: [[0, 0]], bias: [0] },
          evals: [],
        }),
      } as Response;
    }
    throw new Error("unexpected");
  }) as typeof fetch;
  global.fetch = fetchMock;

  render(
    <MlpTrainerPage
      apiBase="http://127.0.0.1:8000"
      datasetName="xor"
      customConfig={{ rows: 1, cols: 2, samples: [] }}
      customApplied={true}
    />,
  );

  await waitFor(() => expect(screen.getByText("MLP (2-layer) training")).toBeInTheDocument());
});
