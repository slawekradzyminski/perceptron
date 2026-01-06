import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { expect, test, vi, describe, beforeEach } from "vitest";
import { DeepPage } from "./DeepPage";

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
  samples: [{ x: [0.5, 0.5], y: 0 }, { x: [-0.5, -0.5], y: 1 }],
};

const mockBoundary = {
  resolution: 10,
  predictions: Array(10).fill(null).map(() => Array(10).fill(0)),
  region_ids: Array(10).fill(null).map(() => Array(10).fill(0)),
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

describe("DeepPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  test("renders Deep Learning Lab header", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.getByText("Deep Learning Lab")).toBeInTheDocument();
    expect(screen.getByText(/Geometry of Depth/)).toBeInTheDocument();
  });

  test("renders control buttons", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByText("Step (S)")).toBeInTheDocument();
    expect(screen.getByText("Train Epoch")).toBeInTheDocument();
    expect(screen.getByText("Train ×10")).toBeInTheDocument();
    expect(screen.getByText("Reset (R)")).toBeInTheDocument();
  });

  test("renders stats card with metrics", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByText("Training Stats")).toBeInTheDocument();
    expect(screen.getByText("Epoch")).toBeInTheDocument();
    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("Loss")).toBeInTheDocument();
    expect(screen.getByText("Accuracy")).toBeInTheDocument();
    expect(screen.getByText("Region Analysis")).toBeInTheDocument();
    expect(screen.getByText("Actual Regions")).toBeInTheDocument();
  });

  test("renders architecture controls", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByText("Architecture")).toBeInTheDocument();
    expect(screen.getByText("Dataset")).toBeInTheDocument();
    expect(screen.getByText("Depth (layers)")).toBeInTheDocument();
    expect(screen.getByText("Width (neurons)")).toBeInTheDocument();
    expect(screen.getByText("Learning Rate")).toBeInTheDocument();
    expect(screen.getByText("Seed")).toBeInTheDocument();
  });

  test("renders comparison table section", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByText(/Architecture Comparison/)).toBeInTheDocument();
    expect(screen.getByText(/Exercise 4.9/)).toBeInTheDocument();
  });

  test("step button is clickable after loading", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.endsWith("/deep/step")) {
        return { ok: true, json: async () => mockStepResponse } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    // Wait for loading to complete - button should become enabled
    await waitFor(() => {
      const stepButton = screen.getByText("Step (S)");
      expect(stepButton).not.toBeDisabled();
    }, { timeout: 3000 });

    const stepButton = screen.getByText("Step (S)");
    expect(stepButton).toBeInTheDocument();

    // Click should not throw
    fireEvent.click(stepButton);
  });

  test("dataset select changes dataset", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "xor" } });

    await waitFor(() => {
      const resetCalls = fetchMock.mock.calls.filter((call) =>
        typeof call[0] === "string"
          ? call[0].endsWith("/deep/reset")
          : (call[0] as Request).url.endsWith("/deep/reset"),
      );
      expect(resetCalls.length).toBeGreaterThan(1); // At least initial + dataset change
    });
  });

  test("show tiling checkbox toggles", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    const tilingCheckbox = screen.getByLabelText("Show Region Tiling");
    expect(tilingCheckbox).not.toBeChecked();

    fireEvent.click(tilingCheckbox);
    expect(tilingCheckbox).toBeChecked();
  });

  test("displays error when API fails", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return {
          ok: false,
          status: 500,
          text: async () => "Server error",
        } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => {
      expect(screen.getByText(/API error/)).toBeInTheDocument();
    });
  });

  test("renders canvas element", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    const canvas = document.querySelector("canvas");
    expect(canvas).toBeInTheDocument();
  });

  test("add to comparison button is present", async () => {
    const fetchMock = vi.fn(async (url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.endsWith("/deep/reset")) {
        return { ok: true, json: async () => mockState } as Response;
      }
      if (target.includes("/deep/boundary")) {
        return { ok: true, json: async () => mockBoundary } as Response;
      }
      return { ok: false } as Response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    render(<DeepPage apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(screen.getByText("Add to Comparison")).toBeInTheDocument();
  });
});

