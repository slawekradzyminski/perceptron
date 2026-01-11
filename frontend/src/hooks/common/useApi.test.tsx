import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useApi, useApiResource } from "./useApi";

describe("useApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("makes a successful GET request", async () => {
    const mockData = { id: 1, name: "test" };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockData,
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() => useApi("http://127.0.0.1:8000"));

    let responseData: unknown = null;
    await act(async () => {
      responseData = await result.current.get("/test");
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/test",
      expect.objectContaining({
        method: "GET",
        headers: { "Content-Type": "application/json" },
      }),
    );
    expect(responseData).toEqual(mockData);
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("makes a successful POST request with body", async () => {
    const mockData = { success: true };
    const requestBody = { name: "test" };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockData,
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() => useApi("http://127.0.0.1:8000"));

    let responseData: unknown = null;
    await act(async () => {
      responseData = await result.current.post("/test", requestBody);
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/test",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      }),
    );
    expect(responseData).toEqual(mockData);
  });

  it("handles API errors", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: "Bad request" }),
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() => useApi("http://127.0.0.1:8000"));

    let responseData: unknown = null;
    await act(async () => {
      responseData = await result.current.get("/test");
    });

    expect(responseData).toBeNull();
    expect(result.current.error).toBe("Bad request");
  });

  it("handles network errors", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("Network error"));
    global.fetch = fetchMock;

    const { result } = renderHook(() => useApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.get("/test");
    });

    expect(result.current.error).toContain("unreachable");
  });

  it("sets loading state correctly", async () => {
    let resolvePromise: (value: unknown) => void;
    const fetchMock = vi.fn().mockImplementation(
      () =>
        new Promise((resolve) => {
          resolvePromise = resolve;
        }),
    );
    global.fetch = fetchMock;

    const { result } = renderHook(() => useApi("http://127.0.0.1:8000"));

    expect(result.current.loading).toBe(false);

    act(() => {
      result.current.get("/test");
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(true);
    });

    await act(async () => {
      resolvePromise!({
        ok: true,
        json: async () => ({ data: "test" }),
      });
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
  });

  it("clears error when clearError is called", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ detail: "Server error" }),
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() => useApi("http://127.0.0.1:8000"));

    await act(async () => {
      await result.current.get("/test");
    });

    expect(result.current.error).not.toBeNull();

    act(() => {
      result.current.clearError();
    });

    expect(result.current.error).toBeNull();
  });
});

describe("useApiResource", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("fetches and stores data", async () => {
    const mockData = { state: "loaded" };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockData,
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useApiResource("http://127.0.0.1:8000", "/state"),
    );

    expect(result.current.data).toBeNull();

    await act(async () => {
      await result.current.fetch();
    });

    expect(result.current.data).toEqual(mockData);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/state",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      }),
    );
  });

  it("handles errors", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Not found" }),
    });
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useApiResource("http://127.0.0.1:8000", "/missing"),
    );

    await act(async () => {
      await result.current.fetch();
    });

    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe("Not found");
  });
});
