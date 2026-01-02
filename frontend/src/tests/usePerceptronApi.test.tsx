import { renderHook, act } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { usePerceptronApi } from "../hooks/usePerceptronApi";

function mockFetchOnce(payload: any) {
  return vi.fn(async () => ({
    ok: true,
    json: async () => payload,
  })) as typeof fetch;
}

test("reset updates state", async () => {
  const fetchMock = mockFetchOnce({
    dataset: "or",
    w: [0, 0],
    b: 0,
    idx: 0,
    next_x: [-1, -1],
    next_y: -1,
    grid_rows: 1,
    grid_cols: 2,
    sample_count: 4,
  });
  global.fetch = fetchMock;

  const { result } = renderHook(() => usePerceptronApi());
  await act(async () => {
    await result.current.reset({
      apiBase: "http://127.0.0.1:8000",
      lr: 1,
      customConfig: { rows: 2, cols: 2, samples: [] },
    });
  });

  expect(result.current.state.datasetName).toBe("or");
  expect(result.current.state.gridRows).toBe(1);
});

test("step updates lastStep", async () => {
  const fetchMock = mockFetchOnce({
    dataset: "or",
    w: [1, 0],
    b: 1,
    idx: 1,
    x: [-1, 1],
    y: 1,
    score: 1,
    pred: 1,
    mistake: false,
    delta_w: [0, 0],
    delta_b: 0,
    lr: 1,
    next_x: [1, 1],
    next_y: 1,
    grid_rows: 1,
    grid_cols: 2,
    sample_count: 4,
  });
  global.fetch = fetchMock;

  const { result } = renderHook(() => usePerceptronApi({
    datasetName: "or",
    gridRows: 1,
    gridCols: 2,
    sampleCount: 4,
    sampleIdx: 0,
    w: [0, 0],
    b: 0,
    lastStep: null,
    nextInput: null,
    apiError: null,
    customApplied: true,
  }));

  await act(async () => {
    await result.current.step({
      apiBase: "http://127.0.0.1:8000",
      lr: 1,
      customConfig: { rows: 2, cols: 2, samples: [] },
    });
  });

  expect(result.current.state.lastStep?.score).toBe(1);
});
