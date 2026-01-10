import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MlpBoundaryPanel } from "./MlpBoundaryPanel";
import type { MlpTrainerSnapshot } from "../../types";

vi.mock("../../utils/visuals", () => ({
  drawMlpBoundary: vi.fn(),
}));

const mockSnapshot: MlpTrainerSnapshot = {
  dataset: "xor",
  grid_rows: 1,
  grid_cols: 2,
  hidden_dim: 2,
  lr: 0.1,
  seed: 1,
  idx: 0,
  sample_count: 1,
  next_x: [0, 0],
  next_y: 1,
  hidden: {
    weights: [[0.1, 0.2], [0.3, 0.4]],
    bias: [0.1, 0.2],
    templates: [[[0]], [[0]]],
  },
  output: {
    weights: [[0.2, 0.3]],
    bias: [0.1],
  },
  evals: [{ x: [0, 0], y: 1, p_hat: 0.6, pred: 1 }],
};

describe("MlpBoundaryPanel", () => {
  beforeEach(() => {
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      clearRect: vi.fn(),
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
  });

  it("renders boundary legend when 2D", () => {
    render(<MlpBoundaryPanel snapshot={mockSnapshot} />);

    expect(screen.getByText("2D Boundary")).toBeInTheDocument();
    expect(screen.getByText("Output boundary")).toBeInTheDocument();
  });

  it("shows placeholder when not 2D", () => {
    render(
      <MlpBoundaryPanel
        snapshot={{ ...mockSnapshot, grid_rows: 1, grid_cols: 3 }}
      />,
    );

    expect(
      screen.getByText("Boundary is available only for 2D inputs."),
    ).toBeInTheDocument();
  });
});
