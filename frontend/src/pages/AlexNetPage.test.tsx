import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { expect, test, vi, describe, beforeEach } from "vitest";
import { AlexNetPage } from "./AlexNetPage";

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
  count: 4,
  layer: 1,
  layer_info: { name: "Conv1", filters: 64, kernel_size: 11, in_channels: 3 },
  filters: [
    { index: 0, kernel_size: 11, image: "dGVzdA==" },
    { index: 1, kernel_size: 11, image: "dGVzdA==" },
    { index: 2, kernel_size: 11, image: "dGVzdA==" },
    { index: 3, kernel_size: 11, image: "dGVzdA==" },
  ],
};

const mockActivations = {
  rows: 8,
  cols: 8,
  count: 4,
  layer: 1,
  layer_info: { name: "Conv1", filters: 64, kernel_size: 11, in_channels: 3 },
  sample_name: "gradient",
  activations: [
    { index: 0, width: 55, height: 55, max_activation: 1.5, image: "dGVzdA==" },
    { index: 1, width: 55, height: 55, max_activation: 1.2, image: "dGVzdA==" },
    { index: 2, width: 55, height: 55, max_activation: 0.8, image: "dGVzdA==" },
    { index: 3, width: 55, height: 55, max_activation: 0.5, image: "dGVzdA==" },
  ],
};

describe("AlexNetPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("renders AlexNet Explorer header", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/alexnet/state")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/alexnet/filters")) {
        return { ok: true, json: async () => mockFilters } as Response;
      }
      if (target.includes("/alexnet/activations")) {
        return { ok: true, json: async () => mockActivations } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<AlexNetPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.getByText("AlexNet Explorer")).toBeInTheDocument();
    expect(screen.getByText(/Visualizing Convolutional/)).toBeInTheDocument();
  });

  test("renders layer buttons", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/alexnet/state")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/alexnet/filters")) {
        return { ok: true, json: async () => mockFilters } as Response;
      }
      if (target.includes("/alexnet/activations")) {
        return { ok: true, json: async () => mockActivations } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<AlexNetPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByRole("button", { name: "1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "2" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "3" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "4" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "5" })).toBeInTheDocument();
  });

  test("renders sample buttons", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/alexnet/state")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/alexnet/filters")) {
        return { ok: true, json: async () => mockFilters } as Response;
      }
      if (target.includes("/alexnet/activations")) {
        return { ok: true, json: async () => mockActivations } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<AlexNetPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByRole("button", { name: "gradient" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "checkerboard" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "noise" })).toBeInTheDocument();
  });

  test("renders filter grid", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/alexnet/state")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/alexnet/filters")) {
        return { ok: true, json: async () => mockFilters } as Response;
      }
      if (target.includes("/alexnet/activations")) {
        return { ok: true, json: async () => mockActivations } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<AlexNetPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => {
      expect(screen.getByText(/Pattern Detectors/)).toBeInTheDocument();
    });

    // Filter images should be present
    const filterImages = document.querySelectorAll(".filter-cell img");
    expect(filterImages.length).toBe(4);
  });

  test("renders activation grid", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/alexnet/state")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/alexnet/filters")) {
        return { ok: true, json: async () => mockFilters } as Response;
      }
      if (target.includes("/alexnet/activations")) {
        return { ok: true, json: async () => mockActivations } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<AlexNetPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => {
      expect(screen.getByText(/Activation Maps/)).toBeInTheDocument();
    });

    const activationImages = document.querySelectorAll(".activation-cell img");
    expect(activationImages.length).toBe(4);
  });

  test("layer change triggers API call", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/alexnet/state")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/alexnet/layer/")) {
        return { ok: true, json: async () => ({ ...mockState, current_layer: 2 }) } as Response;
      }
      if (target.includes("/alexnet/filters")) {
        return { ok: true, json: async () => mockFilters } as Response;
      }
      if (target.includes("/alexnet/activations")) {
        return { ok: true, json: async () => mockActivations } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<AlexNetPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    const layer2Button = screen.getByRole("button", { name: "2" });
    fireEvent.click(layer2Button);

    await waitFor(() => {
      const calls = fetchMock.mock.calls.filter((call) => {
        const url = typeof call[0] === "string" ? call[0] : (call[0] as Request).url;
        return url.includes("/alexnet/layer/2");
      });
      expect(calls.length).toBeGreaterThan(0);
    });
  });

  test("shows model info", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/alexnet/state")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/alexnet/filters")) {
        return { ok: true, json: async () => mockFilters } as Response;
      }
      if (target.includes("/alexnet/activations")) {
        return { ok: true, json: async () => mockActivations } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<AlexNetPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => {
      expect(screen.getByText(/61.1M/)).toBeInTheDocument();
    });
  });

  test("displays error on API failure", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: false,
      status: 500,
      json: async () => ({ detail: "Server error" }),
    })) as unknown as typeof fetch;
    global.fetch = fetchMock;

    render(<AlexNetPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => {
      expect(screen.getByText(/Server error/)).toBeInTheDocument();
    });
  });
});

