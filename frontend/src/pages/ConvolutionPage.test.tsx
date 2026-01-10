import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ConvolutionPage } from "./ConvolutionPage";

const mockState = {
  image: Array(8).fill(Array(8).fill(128)),
  image_name: "cross",
  image_size: 8,
  kernel: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  kernel_name: "identity",
  padding: 1,
  stride: 1,
  input_shape: [8, 8] as [number, number],
  kernel_shape: [3, 3] as [number, number],
  output_shape: [8, 8] as [number, number],
  activation_map: Array(8).fill(Array(8).fill(100)),
  activation_map_normalized: Array(8).fill(Array(8).fill(128)),
  image_base64: "abc123",
  activation_base64: "def456",
};

const mockPresets = {
  images: ["gradient", "checkerboard", "cross"],
  kernels: ["identity", "blur", "sharpen"],
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

describe("ConvolutionPage", () => {
  const apiBase = "http://localhost:8000";

  beforeEach(() => {
    vi.resetAllMocks();
    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes("/conv/digit/load")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            model_ready: true,
            accuracy: 0.98,
          }),
        });
      }
      if (url.includes("/conv/state")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockState),
        });
      }
      if (url.includes("/conv/presets")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockPresets),
        });
      }
      if (url.includes("/conv/step")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockStep),
        });
      }
      if (url.includes("/conv/image/")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ ...mockState, image_name: "gradient" }),
        });
      }
      if (url.includes("/conv/kernel")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ ...mockState, kernel_name: "blur" }),
        });
      }
      if (url.includes("/conv/params")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ ...mockState, padding: 2 }),
        });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockState),
      });
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the page header", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("Convolution Lab")).toBeInTheDocument();
    });
  });

  it("displays input image section", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("Input Image")).toBeInTheDocument();
    });
  });

  it("displays activation map section", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("Activation Map")).toBeInTheDocument();
    });
  });

  it("displays kernel preset buttons", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("identity")).toBeInTheDocument();
      expect(screen.getByText("blur")).toBeInTheDocument();
    });
  });

  it("displays sample image buttons", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("gradient")).toBeInTheDocument();
      expect(screen.getByText("checkerboard")).toBeInTheDocument();
    });
  });

  it("displays CNN education section", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(
        screen.getByText("How Convolutional Neural Networks Work")
      ).toBeInTheDocument();
    });
  });

  it("displays digit recognition section", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("🔢 Digit Recognition")).toBeInTheDocument();
    });
  });

  it("clicking kernel preset button changes kernel", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("blur")).toBeInTheDocument();
    });

    const blurButton = screen.getByText("blur");
    fireEvent.click(blurButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/conv/kernel"),
        expect.objectContaining({ method: "POST" })
      );
    });
  });

  it("clicking image preset button changes image", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("gradient")).toBeInTheDocument();
    });

    const gradientButton = screen.getByText("gradient");
    fireEvent.click(gradientButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/conv/image/gradient"),
        expect.objectContaining({ method: "POST" })
      );
    });
  });

  it("displays current dimensions", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("Current Dimensions")).toBeInTheDocument();
    });

    // Output dimensions should be shown somewhere
    await waitFor(() => {
      expect(screen.getAllByText(/8×8/).length).toBeGreaterThan(0);
    });
  });

  it("has upload image button", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(screen.getByText("Upload Image")).toBeInTheDocument();
    });
  });

  it("displays step math panel", async () => {
    render(<ConvolutionPage apiBase={apiBase} />);

    await waitFor(() => {
      expect(
        screen.getByText("Hover over activation map to see calculation")
      ).toBeInTheDocument();
    });
  });
});
