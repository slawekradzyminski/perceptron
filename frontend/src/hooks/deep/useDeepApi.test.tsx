import { act, renderHook } from "@testing-library/react";
import { expect, test, vi, describe, beforeEach } from "vitest";
import { useDeepApi } from "./useDeepApi";

const mockState = {
  dataset: "circles",
  dataset_info: { name: "Concentric Circles", description: "Two concentric circles", n_classes: 2 },
  sample_count: 100,
  idx: 0,
  epoch: 0,
  total_steps: 0,
  lr: 0.1,
  architecture: {
    input_dim: 2,
    hidden_dims: [8, 8],
    output_dim: 2,
    depth: 2,
    width: 8,
    param_count: 114,
  },
  metrics: { loss: 0.693, accuracy: 0.5 },
  samples: [{ x: [0.5, 0.5], y: 0 }],
};

const mockBoundary = {
  resolution: 10,
  predictions: Array(10).fill(Array(10).fill(0)),
  region_ids: Array(10).fill(Array(10).fill(0)),
  region_count: 5,
  theoretical_max: 100,
};

const mockStepResponse = {
  ...mockState,
  total_steps: 1,
  metrics: { loss: 0.6, accuracy: 0.55 },
  step_info: {
    batch_size: 1,
    last_loss: 0.6,
    last_correct: true,
    last_prediction: 0,
    grad_norm: 0.1,
  },
};

const mockRegions = {
  count: 5,
  theoretical_max: 100,
  efficiency: 0.05,
};

const mockComparisonEntry = {
  depth: 2,
  width: 8,
  hidden_dims: [8, 8],
  param_count: 114,
  actual_regions: 5,
  theoretical_max: 100,
  accuracy: 0.5,
  loss: 0.693,
  total_steps: 0,
};

describe("useDeepApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("fetchState loads state from API", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockState,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchState();
    });

    expect(result.current.state?.dataset).toBe("circles");
    expect(result.current.state?.architecture.depth).toBe(2);
    expect(result.current.error).toBeNull();
  });

  test("fetchState handles error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchState();
    });

    expect(result.current.state).toBeNull();
    expect(result.current.error).toContain("API error");
  });

  test("fetchState handles network error", async () => {
    const fetchMock = vi.fn(async () => {
      throw new Error("Network error");
    }) as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchState();
    });

    expect(result.current.error).toContain("API unreachable");
  });

  test("reset calls API with options", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo, _options?: RequestInit) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return {
          ok: true,
          json: async () => mockState,
        } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.reset({ dataset: "xor", hidden_dims: [16, 16], lr: 0.5, seed: 123 });
    });

    expect(result.current.state?.dataset).toBe("circles");
    expect(result.current.loading).toBe(false);

    const resetCall = fetchMock.mock.calls.find((call) =>
      typeof call[0] === "string"
        ? call[0].endsWith("/deep/reset")
        : (call[0] as Request).url.endsWith("/deep/reset"),
    );
    expect(resetCall).toBeDefined();
    const body = JSON.parse((resetCall?.[1] as RequestInit)?.body as string);
    expect(body.dataset).toBe("xor");
    expect(body.hidden_dims).toEqual([16, 16]);
    expect(body.lr).toBe(0.5);
    expect(body.seed).toBe(123);
  });

  test("reset clears boundary and history", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockState,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.reset({});
    });

    expect(result.current.boundary).toBeNull();
    expect(result.current.history).toEqual([]);
  });

  test("reset handles API error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 400,
      text: async () => "Invalid parameters",
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.reset({});
    });

    expect(result.current.error).toContain("API error");
  });

  test("step calls API and updates state", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockStepResponse,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.step(1);
    });

    expect(result.current.state?.total_steps).toBe(1);
    expect(result.current.history.length).toBe(1);
    expect(result.current.history[0].loss).toBe(0.6);
    expect(result.current.history[0].accuracy).toBe(0.55);
  });

  test("step handles batch size", async () => {
    const fetchMock = vi.fn(async (_url: RequestInfo, _options?: RequestInit) => ({
      ok: true,
      json: async () => mockStepResponse,
    }));
    global.fetch = fetchMock as unknown as typeof fetch;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.step(10);
    });

    const stepCall = fetchMock.mock.calls.find((call: [RequestInfo, RequestInit?]) =>
      typeof call[0] === "string"
        ? call[0].endsWith("/deep/step")
        : (call[0] as Request).url.endsWith("/deep/step"),
    );
    const body = JSON.parse((stepCall?.[1] as RequestInit | undefined)?.body as string);
    expect(body.batch_size).toBe(10);
  });

  test("step handles API error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.step();
    });

    expect(result.current.error).toContain("API error");
  });

  test("trainEpoch calls API", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockStepResponse,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.trainEpoch();
    });

    expect(result.current.state?.total_steps).toBe(1);
    expect(result.current.history.length).toBe(1);
  });

  test("trainEpoch handles API error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.trainEpoch();
    });

    expect(result.current.error).toContain("API error");
  });

  test("fetchBoundary loads boundary data", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockBoundary,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchBoundary(10);
    });

    expect(result.current.boundary?.region_count).toBe(5);
    expect(result.current.regions?.count).toBe(5);
    expect(result.current.regions?.efficiency).toBe(0.05);
  });

  test("fetchBoundary handles API error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchBoundary();
    });

    expect(result.current.error).toContain("API error");
  });

  test("fetchRegions loads region data", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockRegions,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchRegions(20);
    });

    expect(result.current.regions?.count).toBe(5);
    expect(result.current.regions?.theoretical_max).toBe(100);
  });

  test("addToComparison adds entry", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockComparisonEntry,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.addToComparison();
    });

    expect(result.current.comparison.length).toBe(1);
    expect(result.current.comparison[0].depth).toBe(2);
    expect(result.current.comparison[0].actual_regions).toBe(5);
  });

  test("clearComparison clears entries", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockComparisonEntry,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    // Add an entry first
    await act(async () => {
      await result.current.addToComparison();
    });
    expect(result.current.comparison.length).toBe(1);

    // Clear
    await act(async () => {
      await result.current.clearComparison();
    });
    expect(result.current.comparison.length).toBe(0);
  });

  test("loading state is managed correctly", async () => {
    let resolvePromise: (value: Response) => void;
    const fetchPromise = new Promise<Response>((resolve) => {
      resolvePromise = resolve;
    });
    const fetchMock = vi.fn(() => fetchPromise) as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useDeepApi("http://127.0.0.1:8000"));

    expect(result.current.loading).toBe(false);

    let stepPromise: Promise<void>;
    act(() => {
      stepPromise = result.current.step();
    });

    // Loading should be true while waiting
    expect(result.current.loading).toBe(true);

    // Resolve the promise
    await act(async () => {
      resolvePromise!({
        ok: true,
        json: async () => mockStepResponse,
      } as Response);
      await stepPromise;
    });

    expect(result.current.loading).toBe(false);
  });
});

