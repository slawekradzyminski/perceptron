import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useConvApi } from "./useConvApi";

const mockState = {
  image: [[0, 0], [0, 0]],
  image_name: "cross",
  image_size: 8,
  kernel: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  kernel_name: "identity",
  padding: 1,
  stride: 1,
  input_shape: [8, 8],
  kernel_shape: [3, 3],
  output_shape: [8, 8],
  activation_map: [[0]],
  activation_map_normalized: [[128]],
  image_base64: "abc123",
  activation_base64: "def456",
};

const mockPresets = {
  images: ["gradient", "checkerboard"],
  kernels: ["identity", "blur"],
  kernel_weights: {
    identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  },
};

const mockStep = {
  output_row: 0,
  output_col: 0,
  input_row: 0,
  input_col: 0,
  patch: [[0, 0, 0], [0, 100, 0], [0, 0, 0]],
  kernel: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  products: [[0, 0, 0], [0, 100, 0], [0, 0, 0]],
  sum: 100,
  kernel_size: 3,
  padding: 1,
  stride: 1,
};

describe("useConvApi", () => {
  const apiBase = "http://localhost:8000";

  beforeEach(() => {
    vi.resetAllMocks();
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("fetchState populates state on success", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockState,
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    await result.current.fetchState();

    await waitFor(() => {
      expect(result.current.state).toEqual(mockState);
      expect(result.current.error).toBeNull();
    });
  });

  it("fetchState sets error on failure", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => ({ detail: "Server error" }),
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    await result.current.fetchState();

    await waitFor(() => {
      expect(result.current.error).toContain("Server error");
    });
  });

  it("fetchPresets populates presets", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockPresets,
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    await result.current.fetchPresets();

    await waitFor(() => {
      expect(result.current.presets).toEqual(mockPresets);
    });
  });

  it("setImage updates state", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...mockState, image_name: "gradient" }),
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    await result.current.setImage("gradient");

    await waitFor(() => {
      expect(result.current.state?.image_name).toBe("gradient");
    });
  });

  it("setKernel with name updates state", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...mockState, kernel_name: "blur" }),
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    await result.current.setKernel("blur");

    await waitFor(() => {
      expect(result.current.state?.kernel_name).toBe("blur");
    });
  });

  it("setParams updates padding and stride", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...mockState, padding: 2, stride: 2 }),
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    await result.current.setParams(2, 2);

    await waitFor(() => {
      expect(result.current.state?.padding).toBe(2);
      expect(result.current.state?.stride).toBe(2);
    });
  });

  it("getStep populates step result", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStep,
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    await result.current.getStep(0, 0);

    await waitFor(() => {
      expect(result.current.step).toEqual(mockStep);
    });
  });

  it("clearStep clears step result", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => mockStep,
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    await result.current.getStep(0, 0);
    await waitFor(() => {
      expect(result.current.step).toEqual(mockStep);
    });

    result.current.clearStep();

    await waitFor(() => {
      expect(result.current.step).toBeNull();
    });
  });

  it("setImageSize updates image size", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...mockState, image_size: 16 }),
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    await result.current.setImageSize(16);

    await waitFor(() => {
      expect(result.current.state?.image_size).toBe(16);
    });
  });

  it("uploadImage sends FormData", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ ...mockState, image_name: "uploaded" }),
    });

    const { result } = renderHook(() => useConvApi(apiBase));

    const file = new File(["test"], "test.png", { type: "image/png" });
    await result.current.uploadImage(file);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/conv/image/upload"),
        expect.objectContaining({ method: "POST" })
      );
    });
  });
});
