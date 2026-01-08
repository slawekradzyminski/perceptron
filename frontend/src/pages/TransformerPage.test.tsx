import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { expect, test, vi, describe, beforeEach } from "vitest";
import { TransformerPage } from "./TransformerPage";

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
    {
      name: "GPT-4",
      year: 2023,
      params: 1800000000000,
      params_formatted: "1.8T",
      type: "Transformer",
      description: "Multimodal LLM",
      input_size: "32k tokens",
      notable: "Largest model",
    },
  ],
};

const mockGrowth = {
  data: [
    { name: "AlexNet", year: 2012, params: 61000000, log_params: 8, type: "CNN" },
    { name: "GPT-4", year: 2023, params: 1800000000000, log_params: 12, type: "Transformer" },
  ],
  insight: "Models have grown exponentially.",
};

const mockOllamaStatus = {
  ok: true,
  model: "llama3.2:1b",
  base_url: "http://127.0.0.1:11434",
  models: ["llama3.2:1b"],
  model_available: true,
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

const mockEmbedTokens = [
  {
    id: 9906,
    text: "Hello",
    position: 0,
    embedding: Array(768).fill(0.1),
    embedding_preview: [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0],
    min: -0.9,
    max: 1.0,
    mean: 0.05,
    std: 0.5,
  },
  {
    id: 1917,
    text: " world",
    position: 1,
    embedding: Array(768).fill(0.2),
    embedding_preview: [0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.1],
    min: -0.9,
    max: 1.0,
    mean: 0.06,
    std: 0.52,
  },
  {
    id: 0,
    text: "!",
    position: 2,
    embedding: Array(768).fill(0.3),
    embedding_preview: [-0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0, 0.1, 0.2],
    min: -0.9,
    max: 1.0,
    mean: 0.04,
    std: 0.48,
  },
];

const mockEmbed = {
  text: "Hello world!",
  token_count: 3,
  d_model: 768,
  matrix_shape: [3, 768],
  explanation: "Each token is a 768-dimensional vector.",
  tokens: mockEmbedTokens,
  full_matrix: mockEmbedTokens.map(t => t.embedding),
};

describe("TransformerPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("renders Transformer Flow header", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/transformer/ollama-status")) {
        return { ok: true, json: async () => mockOllamaStatus } as Response;
      }
      if (target.includes("/transformer/scale/growth")) {
        return { ok: true, json: async () => mockGrowth } as Response;
      }
      if (target.includes("/transformer/scale")) {
        return { ok: true, json: async () => mockScaleModels } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<TransformerPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.getByText("Transformer Flow")).toBeInTheDocument();
    expect(screen.getByText(/Understanding GPT/)).toBeInTheDocument();
  });

  test("renders tab navigation", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/transformer/ollama-status")) {
        return { ok: true, json: async () => mockOllamaStatus } as Response;
      }
      if (target.includes("/transformer/scale/growth")) {
        return { ok: true, json: async () => mockGrowth } as Response;
      }
      if (target.includes("/transformer/scale")) {
        return { ok: true, json: async () => mockScaleModels } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<TransformerPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByRole("button", { name: "Tokenization" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Block Trace" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Model Scale" })).toBeInTheDocument();
  });

  test("renders text input", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/transformer/ollama-status")) {
        return { ok: true, json: async () => mockOllamaStatus } as Response;
      }
      if (target.includes("/transformer/scale/growth")) {
        return { ok: true, json: async () => mockGrowth } as Response;
      }
      if (target.includes("/transformer/scale")) {
        return { ok: true, json: async () => mockScaleModels } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<TransformerPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeInTheDocument();
    expect(textarea).toHaveValue("The quick brown fox jumps");
  });

  test("tokenize button triggers API call", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/transformer/ollama-status")) {
        return { ok: true, json: async () => mockOllamaStatus } as Response;
      }
      if (target.includes("/transformer/scale/growth")) {
        return { ok: true, json: async () => mockGrowth } as Response;
      }
      if (target.includes("/transformer/scale")) {
        return { ok: true, json: async () => mockScaleModels } as Response;
      }
      if (target.includes("/transformer/tokenize")) {
        return { ok: true, json: async () => mockTokens } as Response;
      }
      if (target.includes("/transformer/embed")) {
        return { ok: true, json: async () => mockEmbed } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<TransformerPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    const tokenizeButton = screen.getByRole("button", { name: "Tokenize & Embed" });
    fireEvent.click(tokenizeButton);

    await waitFor(() => {
      const calls = fetchMock.mock.calls.filter((call) => {
        const url = typeof call[0] === "string" ? call[0] : (call[0] as Request).url;
        return url.includes("/transformer/tokenize");
      });
      expect(calls.length).toBeGreaterThan(0);
    });
  });

  test("switching tabs changes content", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/transformer/ollama-status")) {
        return { ok: true, json: async () => mockOllamaStatus } as Response;
      }
      if (target.includes("/transformer/scale/growth")) {
        return { ok: true, json: async () => mockGrowth } as Response;
      }
      if (target.includes("/transformer/scale")) {
        return { ok: true, json: async () => mockScaleModels } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<TransformerPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    // Click on Scale tab
    const scaleTab = screen.getByRole("button", { name: "Model Scale" });
    fireEvent.click(scaleTab);

    await waitFor(() => {
      expect(screen.getByText("All Models")).toBeInTheDocument();
    });
  });

  test("renders scale table with models", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/transformer/ollama-status")) {
        return { ok: true, json: async () => mockOllamaStatus } as Response;
      }
      if (target.includes("/transformer/scale/growth")) {
        return { ok: true, json: async () => mockGrowth } as Response;
      }
      if (target.includes("/transformer/scale")) {
        return { ok: true, json: async () => mockScaleModels } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<TransformerPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    // Click on Scale tab
    const scaleTab = screen.getByRole("button", { name: "Model Scale" });
    fireEvent.click(scaleTab);

    await waitFor(() => {
      // Check for table headers and content
      expect(screen.getByText("All Models")).toBeInTheDocument();
      expect(screen.getByRole("table")).toBeInTheDocument();
    });
  });

  test("renders growth insight", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/transformer/ollama-status")) {
        return { ok: true, json: async () => mockOllamaStatus } as Response;
      }
      if (target.includes("/transformer/scale/growth")) {
        return { ok: true, json: async () => mockGrowth } as Response;
      }
      if (target.includes("/transformer/scale")) {
        return { ok: true, json: async () => mockScaleModels } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<TransformerPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    // Click on Scale tab
    const scaleTab = screen.getByRole("button", { name: "Model Scale" });
    fireEvent.click(scaleTab);

    await waitFor(() => {
      expect(screen.getByText(/exponentially/)).toBeInTheDocument();
    });
  });

  test("text input is editable", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/transformer/ollama-status")) {
        return { ok: true, json: async () => mockOllamaStatus } as Response;
      }
      if (target.includes("/transformer/scale/growth")) {
        return { ok: true, json: async () => mockGrowth } as Response;
      }
      if (target.includes("/transformer/scale")) {
        return { ok: true, json: async () => mockScaleModels } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<TransformerPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    const textarea = screen.getByRole("textbox");
    fireEvent.change(textarea, { target: { value: "New text" } });

    expect(textarea).toHaveValue("New text");
  });
});

