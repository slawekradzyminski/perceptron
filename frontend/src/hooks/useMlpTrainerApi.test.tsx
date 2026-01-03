import { act, renderHook } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { useMlpTrainerApi } from "./useMlpTrainerApi";

test("resets and steps MLP", async () => {
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
    if (target.endsWith("/mlp/step")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "xor",
          grid_rows: 1,
          grid_cols: 2,
          hidden_dim: 2,
          lr: 0.5,
          seed: 0,
          idx: 1,
          sample_count: 4,
          next_x: [-1, 1],
          next_y: 1,
          hidden: { weights: [[0, 0], [0, 0]], bias: [0, 0], templates: [[[0, 0]], [[0, 0]]] },
          output: { weights: [[0, 0]], bias: [0] },
          evals: [],
          step: {
            dataset: "xor",
            grid_rows: 1,
            grid_cols: 2,
            hidden_dim: 2,
            sample_index: 0,
            sample_count: 4,
            x: [-1, -1],
            y: -1,
            y01: 0,
            loss: 0.5,
            p_hat: 0.5,
            hidden: {
              weights_before: [[0, 0], [0, 0]],
              bias_before: [0, 0],
              weights_after: [[0, 0], [0, 0]],
              bias_after: [0, 0],
              z: [0, 0],
              a: [0, 0],
              templates_before: [[[0, 0]], [[0, 0]]],
              templates_after: [[[0, 0]], [[0, 0]]],
            },
            output: {
              weights_before: [[0, 0]],
              bias_before: [0],
              weights_after: [[0, 0]],
              bias_after: [0],
              z: 0,
              a: 0.5,
            },
            gradients: {
              hidden_W: [[0, 0], [0, 0]],
              hidden_b: [0, 0],
              out_W: [[0, 0]],
              out_b: [0],
              templates: [[[0, 0]], [[0, 0]]],
            },
          },
        }),
      } as Response;
    }
    throw new Error("unexpected");
  }) as typeof fetch;
  global.fetch = fetchMock;

  const { result } = renderHook(() => useMlpTrainerApi("http://127.0.0.1:8000"));

  await act(async () => {
    await result.current.resetWithOptions({ datasetName: "xor", hiddenDim: 2, lr: 0.5, seed: 0 });
  });
  expect(result.current.snapshot?.dataset).toBe("xor");

  await act(async () => {
    await result.current.step();
  });
  expect(result.current.lastStep?.dataset).toBe("xor");
});
