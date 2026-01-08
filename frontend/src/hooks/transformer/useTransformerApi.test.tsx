import { act, renderHook } from "@testing-library/react";
import { expect, test, vi, describe, beforeEach } from "vitest";
import { useTransformerApi } from "./useTransformerApi";

const mockState = {
  current_text: "",
  current_token_count: 0,
  encoding_name: "cl100k_base",
  vocab_size: 100277,
};

const mockTokens = {
  text: "Hello world!",
  token_count: 3,
  tokens: [
    { id: 9906, text: "Hello", position: 0 },
    { id: 1917, text: " world", position: 1 },
    { id: 0, text: "!", position: 2 },
  ],
};

const mockEmbed = {
  text: "Hello world!",
  token_count: 3,
  d_model: 768,
  matrix_shape: [3, 768] as [number, number],
  explanation: "Each token is converted to a 768-dimensional vector.",
  tokens: mockTokens.tokens,
};

const mockTrace = {
  text: "Hello",
  token_count: 1,
  d_model: 768,
  n_blocks: 12,
  blocks: [
    {
      block: 1,
      input_shape: [1, 768] as [number, number],
      output_shape: [1, 768] as [number, number],
      operations: ["Attention", "FFN"],
    },
  ],
  final_shape: [1, 768] as [number, number],
  explanation: "The matrix passes through 12 blocks.",
};

const mockChatStreamData = "data: {\"token\":\"Hello\",\"done\":false}\n\ndata: {\"token\":\"!\",\"done\":false}\n\ndata: {\"token\":\"\",\"done\":true}\n\n";

const mockScaleModels = {
  models: [
    {
      name: "AlexNet",
      year: 2012,
      params: 61000000,
      params_formatted: "61.0M",
      type: "CNN",
      description: "ImageNet breakthrough",
      input_size: "224×224",
      notable: "ReLU, dropout",
    },
  ],
};

const mockComparison = {
  model1: mockScaleModels.models[0],
  model2: { ...mockScaleModels.models[0], name: "GPT-4", params: 1800000000000 },
  ratio: 29508.2,
  ratio_formatted: "29508.2×",
  year_gap: 11,
  explanation: "GPT-4 is much larger than AlexNet",
};

const mockGrowth = {
  data: [{ name: "AlexNet", year: 2012, params: 61000000, log_params: 8, type: "CNN" }],
  insight: "Models have grown exponentially.",
};

describe("useTransformerApi", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("fetchState loads state from API", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockState,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.fetchState();
    });

    expect(result.current.state?.encoding_name).toBe("cl100k_base");
    expect(result.current.error).toBeNull();
  });

  test("fetchState handles error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.fetchState();
    });

    expect(result.current.error).toContain("API error");
  });

  test("tokenize calls API with text", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockTokens,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.tokenize("Hello world!");
    });

    expect(result.current.tokens?.text).toBe("Hello world!");
    expect(result.current.tokens?.token_count).toBe(3);
  });

  test("tokenize handles API error", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 400,
      json: async () => ({ detail: "Missing text" }),
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.tokenize("");
    });

    expect(result.current.error).toContain("Missing text");
  });

  test("getEmbedInfo loads embedding info", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockEmbed,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.getEmbedInfo("Hello world!");
    });

    expect(result.current.embedInfo?.d_model).toBe(768);
    expect(result.current.embedInfo?.matrix_shape).toEqual([3, 768]);
  });

  test("getTrace loads block trace", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockTrace,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.getTrace("Hello", 12);
    });

    expect(result.current.trace?.n_blocks).toBe(12);
    expect(result.current.trace?.blocks.length).toBeGreaterThan(0);
  });

  test("simulateGeneration loads generation result", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      body: {
        getReader: () => ({
          read: vi.fn()
            .mockResolvedValueOnce({ done: false, value: new TextEncoder().encode(mockChatStreamData) })
            .mockResolvedValueOnce({ done: true, value: undefined }),
        }),
      },
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.sendChat("Hello");
    });

    // Chat messages should include user message and assistant response
    expect(result.current.chatMessages.length).toBe(2);
    expect(result.current.chatMessages[0].role).toBe("user");
    expect(result.current.chatMessages[0].content).toBe("Hello");
  });

  test("fetchScaleModels loads models", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockScaleModels,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.fetchScaleModels();
    });

    expect(result.current.scaleModels.length).toBe(1);
    expect(result.current.scaleModels[0].name).toBe("AlexNet");
  });

  test("compareModels loads comparison", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockComparison,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.compareModels("AlexNet", "GPT-4");
    });

    expect(result.current.comparison?.ratio).toBeGreaterThan(1);
  });

  test("fetchGrowth loads growth data", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => mockGrowth,
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    await act(async () => {
      await result.current.fetchGrowth();
    });

    expect(result.current.growth?.data.length).toBe(1);
    expect(result.current.growth?.insight).toContain("exponentially");
  });

  test("loading state is managed correctly", async () => {
    let resolvePromise: (value: Response) => void;
    const fetchPromise = new Promise<Response>((resolve) => {
      resolvePromise = resolve;
    });
    const fetchMock = vi.fn(() => fetchPromise) as typeof fetch;
    global.fetch = fetchMock;

    const { result } = renderHook(() =>
      useTransformerApi("http://127.0.0.1:8000"),
    );

    expect(result.current.loading).toBe(false);

    let tokenizePromise: Promise<void>;
    act(() => {
      tokenizePromise = result.current.tokenize("Hello");
    });

    expect(result.current.loading).toBe(true);

    await act(async () => {
      resolvePromise!({
        ok: true,
        json: async () => mockTokens,
      } as Response);
      await tokenizePromise;
    });

    expect(result.current.loading).toBe(false);
  });
});

