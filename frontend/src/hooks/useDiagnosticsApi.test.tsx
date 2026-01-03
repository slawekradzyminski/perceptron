import { act, renderHook } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { useDiagnosticsApi } from "./useDiagnosticsApi";

test("fetches error surface and mlp internals", async () => {
  const fetchMock = vi.fn(async (url: RequestInfo) => {
    if (typeof url === "string" && url.endsWith("/error-surface")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "or",
          grid_rows: 1,
          grid_cols: 2,
          steps: 3,
          w_range: [-1, 1],
          bias: 0,
          sample_count: 4,
          grid: [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
          ],
        }),
      } as Response;
    }
    return {
      ok: true,
      json: async () => ({
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
      }),
    } as Response;
  }) as typeof fetch;
  global.fetch = fetchMock;

  const { result } = renderHook(() =>
    useDiagnosticsApi("http://127.0.0.1:8000", "or", {
      rows: 2,
      cols: 2,
      samples: [{ grid: [[-1, -1], [-1, -1]], y: -1 }],
    }),
  );

  await act(async () => {
    await result.current.fetchErrorSurface({ steps: 3, wMin: -1, wMax: 1, bias: 0 });
    await result.current.fetchMlpInternals({ hiddenDim: 2, sampleIndex: 0, lr: 0.5, seed: 0 });
  });

  expect(result.current.errorSurface?.steps).toBe(3);
  expect(result.current.mlpInternals?.hidden_dim).toBe(2);
});
