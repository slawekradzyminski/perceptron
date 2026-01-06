import { render, screen } from "@testing-library/react";
import { expect, test, describe, vi, beforeEach } from "vitest";
import { DeepLossChart } from "./DeepLossChart";

const mockHistory = [
  { step: 1, epoch: 0, loss: 0.7, accuracy: 0.5 },
  { step: 2, epoch: 0, loss: 0.6, accuracy: 0.55 },
  { step: 3, epoch: 0, loss: 0.5, accuracy: 0.6 },
  { step: 4, epoch: 1, loss: 0.4, accuracy: 0.7 },
  { step: 5, epoch: 1, loss: 0.35, accuracy: 0.75 },
];

describe("DeepLossChart", () => {
  beforeEach(() => {
    // Mock canvas context with all required methods
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      fillStyle: "",
      strokeStyle: "",
      lineWidth: 0,
      font: "",
      textAlign: "",
      fillRect: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fillText: vi.fn(),
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });
  test("renders empty state when history is empty", () => {
    render(<DeepLossChart history={[]} />);
    expect(screen.getByText("Train to see loss/accuracy curves")).toBeInTheDocument();
  });

  test("renders empty state when history has only one entry", () => {
    render(<DeepLossChart history={[{ step: 1, epoch: 0, loss: 0.7, accuracy: 0.5 }]} />);
    expect(screen.getByText("Train to see loss/accuracy curves")).toBeInTheDocument();
  });

  test("renders canvas when history has multiple entries", () => {
    render(<DeepLossChart history={mockHistory} />);

    const canvas = document.querySelector("canvas");
    expect(canvas).toBeInTheDocument();
  });

  test("renders legend when history has data", () => {
    render(<DeepLossChart history={mockHistory} />);

    expect(screen.getByText("■ Loss")).toBeInTheDocument();
    expect(screen.getByText("■ Accuracy")).toBeInTheDocument();
  });

  test("does not render legend when empty", () => {
    render(<DeepLossChart history={[]} />);

    expect(screen.queryByText("■ Loss")).not.toBeInTheDocument();
    expect(screen.queryByText("■ Accuracy")).not.toBeInTheDocument();
  });

  test("canvas has correct dimensions", () => {
    render(<DeepLossChart history={mockHistory} />);

    const canvas = document.querySelector("canvas");
    expect(canvas).toHaveAttribute("width", "280");
    expect(canvas).toHaveAttribute("height", "140");
  });

  test("renders with two entries (minimum for chart)", () => {
    const minHistory = [
      { step: 1, epoch: 0, loss: 0.7, accuracy: 0.5 },
      { step: 2, epoch: 0, loss: 0.6, accuracy: 0.55 },
    ];
    render(<DeepLossChart history={minHistory} />);

    const canvas = document.querySelector("canvas");
    expect(canvas).toBeInTheDocument();
    expect(screen.getByText("■ Loss")).toBeInTheDocument();
  });

  test("updates when history changes", () => {
    const { rerender } = render(<DeepLossChart history={[]} />);
    expect(screen.getByText("Train to see loss/accuracy curves")).toBeInTheDocument();

    rerender(<DeepLossChart history={mockHistory} />);

    const canvas = document.querySelector("canvas");
    expect(canvas).toBeInTheDocument();
    expect(screen.queryByText("Train to see loss/accuracy curves")).not.toBeInTheDocument();
  });
});

