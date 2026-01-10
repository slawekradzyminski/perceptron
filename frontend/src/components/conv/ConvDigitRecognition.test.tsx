import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ConvDigitRecognition } from "./ConvDigitRecognition";

const mockResult = {
  activations: [
    { name: "Conv1", shape: [1, 28, 28], image: "dGVzdA==" },
  ],
  prediction: 7,
  confidence: 0.85,
  probabilities: [
    { digit: 7, probability: 0.85 },
    { digit: 1, probability: 0.05 },
  ],
};

class MockFileReader {
  public onload: ((event: { target: { result: string } }) => void) | null = null;
  readAsDataURL() {
    this.onload?.({ target: { result: "data:image/png;base64,preview" } });
  }
}

describe("ConvDigitRecognition", () => {
  const apiBase = "http://localhost:8000";

  beforeEach(() => {
    vi.resetAllMocks();
    global.fetch = vi.fn((url: RequestInfo) => {
      const target = typeof url === "string" ? url : url.url;
      if (target.includes("/conv/digit/load")) {
        return Promise.resolve({ ok: true });
      }
      if (target.includes("/conv/digit/recognize")) {
        return Promise.resolve({
          ok: true,
          json: async () => mockResult,
        });
      }
      return Promise.resolve({ ok: false });
    }) as unknown as typeof fetch;
    global.FileReader = MockFileReader as unknown as typeof FileReader;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("uploads an image and renders results", async () => {
    render(<ConvDigitRecognition apiBase={apiBase} />);

    expect(screen.getByText("🔢 Digit Recognition")).toBeInTheDocument();

    const input = document.querySelector("input[type=\"file\"]") as HTMLInputElement;
    const file = new File(["test"], "digit.png", { type: "image/png" });

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/conv/digit/recognize"),
        expect.objectContaining({ method: "POST" }),
      );
    });

    await waitFor(() => {
      expect(screen.getByText("Predicted Digit:")).toBeInTheDocument();
      expect(screen.getAllByText("7").length).toBeGreaterThan(0);
    });
  });
});
