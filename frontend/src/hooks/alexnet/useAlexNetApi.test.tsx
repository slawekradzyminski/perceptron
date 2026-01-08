import { act, renderHook } from "@testing-library/react";
import { expect, test, vi, describe, beforeEach } from "vitest";
import { useAlexNetApi } from "./useAlexNetApi";

const mockState = {
  current_layer: 1,
  sample_images: ["gradient", "checkerboard", "noise"],
  layers: [
    { layer: 1, name: "Conv1", filters: 64, kernel_size: 11, in_channels: 3 },
    { layer: 2, name: "Conv2", filters: 192, kernel_size: 5, in_channels: 64 },
  ],
  param_count: 61100840,
};

const mockFilters = {
  rows: 8,
  cols: 8,
  count: 64,
  layer: 1,
  layer_info: { name: "Conv1", filters: 64, kernel_size: 11, in_channels: 3 },
  filters: [{ index: 0, kernel_size: 11, image: "base64data" }],
};

const mockActivations = {
  rows: 8,
  cols: 8,
  count: 64,
  layer: 1,
  layer_info: { name: "Conv1", filters: 64, kernel_size: 11, in_channels: 3 },
  sample_name: "gradient",
  activations: [
    { index: 0, width: 55, height: 55, max_activation: 1.5, image: "base64data" },
  ],
};

describe("useAlexNetApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("fetchState loads state from API", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockState,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchState();
    });

    expect(result.current.state?.current_layer).toBe(1);
    expect(result.current.state?.layers.length).toBe(2);
    expect(result.current.error).toBeNull();
  });

  test("fetchState handles error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

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

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchState();
    });

    expect(result.current.error).toContain("API unreachable");
  });

  test("setLayer calls API", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ ...mockState, current_layer: 2 }),
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.setLayer(2);
    });

    expect(result.current.state?.current_layer).toBe(2);
    expect(result.current.loading).toBe(false);
  });

  test("setLayer handles error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 400,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.setLayer(0);
    });

    expect(result.current.error).toContain("API error");
  });

  test("fetchFilters loads filters", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockFilters,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchFilters(1, 64);
    });

    expect(result.current.filters?.layer).toBe(1);
    expect(result.current.filters?.count).toBe(64);
  });

  test("fetchFilters handles error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchFilters();
    });

    expect(result.current.error).toContain("API error");
  });

  test("fetchActivations loads activations", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockActivations,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchActivations("gradient", 1, 64);
    });

    expect(result.current.activations?.sample_name).toBe("gradient");
    expect(result.current.activations?.layer).toBe(1);
  });

  test("fetchActivations handles error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.fetchActivations();
    });

    expect(result.current.error).toContain("API error");
  });

  test("uploadAndGetActivations posts file", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ ...mockActivations, sample_name: null }),
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    const file = new File(["test"], "test.png", { type: "image/png" });

    await act(async () => {
      await result.current.uploadAndGetActivations(file, 1, 64);
    });

    expect(result.current.activations?.sample_name).toBeNull();
    expect(fetchMock).toHaveBeenCalled();
  });

  test("uploadAndGetActivations handles error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 400,
      text: async () => "Invalid file",
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    const file = new File(["test"], "test.txt", { type: "text/plain" });

    await act(async () => {
      await result.current.uploadAndGetActivations(file);
    });

    expect(result.current.error).toContain("API error");
  });

  test("loading state is managed correctly", async () => {
    let resolvePromise: (value: Response) => void;
    const fetchPromise = new Promise<Response>((resolve) => {
      resolvePromise = resolve;
    });
    const fetchMock = vi.fn(() => fetchPromise) as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() => useAlexNetApi("http://127.0.0.1:8000"));

    expect(result.current.loading).toBe(false);

    let fetchFiltersPromise: Promise<void>;
    act(() => {
      fetchFiltersPromise = result.current.fetchFilters();
    });

    expect(result.current.loading).toBe(true);

    await act(async () => {
      resolvePromise!({
        ok: true,
        json: async () => mockFilters,
      } as Response);
      await fetchFiltersPromise;
    });

    expect(result.current.loading).toBe(false);
  });
});

