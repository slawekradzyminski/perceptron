import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useGdApi } from "./useGdApi";

const mockTokenLosses = {
  examples: [
    {
      id: "1",
      title: "Example",
      description: "desc",
      average: { l1: 0.1, ce: 0.2 },
      rows: [],
    },
  ],
};

const mockStatus = {
  ok: true,
  model: "llama",
};

describe("useGdApi", () => {
  const apiBase = "http://localhost:8000";

  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads token losses and status", async () => {
    global.fetch = vi.fn((url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/gd/token-losses")) {
        return Promise.resolve({ ok: true, json: async () => mockTokenLosses });
      }
      if (target.includes("/gd/ollama-status")) {
        return Promise.resolve({ ok: true, json: async () => mockStatus });
      }
      return Promise.resolve({ ok: false });
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useGdApi(apiBase));

    await waitFor(() => {
      expect(result.current.tokenLosses).toEqual(mockTokenLosses);
      expect(result.current.status).toEqual(mockStatus);
    });
  });

  it("handles fetch errors", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("boom")) as typeof fetch;

    const { result } = renderHook(() => useGdApi(apiBase));

    await waitFor(() => {
      expect(result.current.error).toBe("boom");
      expect(result.current.loading).toBe(false);
    });
  });
});
