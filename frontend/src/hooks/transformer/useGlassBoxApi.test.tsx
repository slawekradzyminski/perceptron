import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useGlassBoxApi } from "./useGlassBoxApi";

const mockAttentionResult = {
  text: "Hello",
  tokens: ["Hello"],
  n_layers: 12,
  n_heads: 12,
  seq_len: 1,
  model_name: "gpt2",
  attentions: [[[[1.0]]]],
  explanation: "Test",
};

const mockLogitLensResult = {
  text: "Hello",
  tokens: ["Hello"],
  n_layers: 12,
  top_k: 5,
  model_name: "gpt2",
  layers: [
    {
      layer: 0,
      layer_name: "Embedding",
      predictions: [{ token: "world", token_id: 1, probability: 0.5 }],
    },
  ],
  explanation: "Test",
};

const mockKvCacheResult = {
  config: {
    context_length: 4096,
    n_layers: 32,
    d_model: 4096,
    n_heads: 32,
    d_head: 128,
    gqa_groups: 8,
    mla_latent_dim: 512,
    bytes_per_param: 2,
  },
  architectures: [
    {
      name: "MHA",
      full_name: "Multi-Head Attention",
      description: "Standard",
      memory_bytes: 2147483648,
      memory_formatted: "2.00 GB",
      ratio_to_mha: 1.0,
      kv_heads: 32,
    },
  ],
  mha_to_mla_savings: 16.0,
  explanation: "Test",
};

describe("useGlassBoxApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("fetches attention patterns", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockAttentionResult,
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useGlassBoxApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.fetchAttention("Hello");
    });

    await waitFor(() => {
      expect(result.current.attention).not.toBeNull();
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/transformer/attention",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ text: "Hello" }),
      }),
    );
    expect(result.current.attention?.model_name).toBe("gpt2");
  });

  it("fetches logit lens", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockLogitLensResult,
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useGlassBoxApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.fetchLogitLens("Hello", 5);
    });

    await waitFor(() => {
      expect(result.current.logitLens).not.toBeNull();
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/transformer/logit-lens",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ text: "Hello", top_k: 5 }),
      }),
    );
    expect(result.current.logitLens?.n_layers).toBe(12);
  });

  it("fetches KV cache comparison", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockKvCacheResult,
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useGlassBoxApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.fetchKvCache({ context_length: 8192 });
    });

    await waitFor(() => {
      expect(result.current.kvCache).not.toBeNull();
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/transformer/kv-cache",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ context_length: 8192 }),
      }),
    );
    expect(result.current.kvCache?.mha_to_mla_savings).toBe(16.0);
  });

  it("handles API errors", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "Bad request" }),
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useGlassBoxApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.fetchAttention("Hello");
    });

    await waitFor(() => {
      expect(result.current.error).not.toBeNull();
    });

    expect(result.current.error).toContain("Bad request");
  });

  it("handles network errors", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("Network error"));
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useGlassBoxApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.fetchAttention("Hello");
    });

    await waitFor(() => {
      expect(result.current.error).not.toBeNull();
    });

    expect(result.current.error).toContain("unreachable");
  });

  it("sets loading state", async () => {
    let resolvePromise: (value: unknown) => void;
    const fetchMock = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePromise = resolve;
        }),
    );
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useGlassBoxApi("http://127.0.0.1:8000"),
    );

    expect(result.current.loading).toBe(false);

    act(() => {
      result.current.fetchAttention("Hello");
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(true);
    });

    await act(async () => {
      resolvePromise!({
        ok: true,
        json: async () => mockAttentionResult,
      });
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
  });
});
