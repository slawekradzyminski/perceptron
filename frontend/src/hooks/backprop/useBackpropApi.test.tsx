import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useBackpropApi } from "./useBackpropApi";
import type { BackpropState, RegressionStep, TinyGpsStep } from "../../types";

const mockState: BackpropState = {
  tinygps: {
    dataset: "europe",
    mode: "1d",
    cities: ["paris"],
    idx: 0,
    lr: 0.1,
    sample_count: 1,
    feature_order: ["lon"],
    sample_order: [0],
    params: { m: [0.1], b: [0.2], M: [] },
    next_sample: { x: [0.2], y: 0, city: "paris", lat: 0, lon: 0 },
    samples: [{ x: [0.2], y: 0, city: "paris", lat: 0, lon: 0 }],
  },
  regression: {
    loss: "mse",
    idx: 0,
    lr: 0.1,
    sample_count: 1,
    params: { m: 1, b: 1 },
    next_sample: { x: 1, y: 2 },
    samples: [{ x: 1, y: 2 }],
    sample_order: [0],
  },
  tinygps_datasets: ["europe"],
  tinygps_city_coords: { paris: [[48.8, 2.3]] },
};

const mockTinyStep: TinyGpsStep = {
  dataset: "europe",
  mode: "1d",
  cities: ["paris"],
  idx: 1,
  sample_idx: 0,
  lr: 0.1,
  feature_order: ["lon"],
  sample: { x: [0.2], y: 0, city: "paris", lat: 0, lon: 0 },
  logits: [0.2],
  probs: [0.8],
  loss: 0.2,
  pred: 0,
  correct: true,
  grads: { m: [0.1], b: [0.2] },
  params_before: { m: [0.1], b: [0.2], M: [] },
  params_after: { m: [0.1], b: [0.2], M: [] },
  metrics_after: { loss: 0.2, accuracy: 1 },
  sample_count: 1,
};

const mockRegressionStep: RegressionStep = {
  loss: "mse",
  idx: 1,
  sample_idx: 0,
  lr: 0.1,
  sample: { x: 1, y: 2 },
  y_hat: 2,
  loss_value: 1,
  grads: { m: 0.5, b: 0.1 },
  params_before: { m: 1, b: 1 },
  params_after: { m: 1.1, b: 1.2 },
  mean_loss: 1,
  sample_count: 1,
};

describe("useBackpropApi", () => {
  const apiBase = "http://localhost:8000";

  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads state on mount", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockState,
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useBackpropApi(apiBase));

    await waitFor(() => {
      expect(result.current.state).toEqual(mockState);
    });
  });

  it("resets tinygps and clears history", async () => {
    global.fetch = vi.fn((url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/backprop/state")) {
        return Promise.resolve({ ok: true, json: async () => mockState });
      }
      if (target.includes("/backprop/tinygps/reset")) {
        return Promise.resolve({ ok: true, json: async () => mockState.tinygps });
      }
      return Promise.resolve({ ok: false });
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useBackpropApi(apiBase));

    await waitFor(() => expect(result.current.state).toEqual(mockState));

    await result.current.resetTinygps("europe", 0.1, { m: [0.1], b: [0.2] });

    await waitFor(() => {
      expect(result.current.tinygpsHistory).toEqual([]);
      expect(result.current.state?.tinygps.dataset).toBe("europe");
    });
  });

  it("steps regression and appends history", async () => {
    global.fetch = vi.fn((url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/backprop/state")) {
        return Promise.resolve({ ok: true, json: async () => mockState });
      }
      if (target.includes("/backprop/regression/step")) {
        return Promise.resolve({ ok: true, json: async () => mockRegressionStep });
      }
      return Promise.resolve({ ok: false });
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useBackpropApi(apiBase));

    await waitFor(() => expect(result.current.state).toEqual(mockState));

    await result.current.stepRegression();

    await waitFor(() => {
      expect(result.current.regressionHistory).toHaveLength(1);
      expect(result.current.regressionHistory[0]?.idx).toBe(1);
    });
  });

  it("steps tinygps and appends history", async () => {
    global.fetch = vi.fn((url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/backprop/state")) {
        return Promise.resolve({ ok: true, json: async () => mockState });
      }
      if (target.includes("/backprop/tinygps/step")) {
        return Promise.resolve({ ok: true, json: async () => mockTinyStep });
      }
      return Promise.resolve({ ok: false });
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useBackpropApi(apiBase));

    await waitFor(() => expect(result.current.state).toEqual(mockState));

    await result.current.stepTinygps();

    await waitFor(() => {
      expect(result.current.tinygpsHistory).toHaveLength(1);
      expect(result.current.tinygpsHistory[0]?.idx).toBe(1);
    });
  });
});
