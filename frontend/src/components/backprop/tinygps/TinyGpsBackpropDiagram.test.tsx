import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { TinyGpsBackpropDiagram } from "./TinyGpsBackpropDiagram";
import type { TinyGpsStep } from "../../../types";

const mockStep: TinyGpsStep = {
  dataset: "europe",
  mode: "1d",
  cities: ["paris", "madrid"],
  idx: 0,
  sample_idx: 0,
  lr: 0.1,
  feature_order: ["lon"],
  sample: { x: [0.2], y: 0, city: "paris", lat: 0, lon: 0 },
  logits: [0.2, -0.1],
  probs: [0.7, 0.3],
  loss: 0.4,
  pred: 0,
  correct: true,
  grads: { m: [0.1, 0.2], b: [0.05, 0.1] },
  params_before: { m: [0.1, -0.2], b: [0.0, 0.1], M: [] },
  params_after: { m: [0.1, -0.2], b: [0.0, 0.1], M: [] },
  metrics_after: { loss: 0.4, accuracy: 1 },
  sample_count: 2,
};

describe("TinyGpsBackpropDiagram", () => {
  it("renders placeholder when no step", () => {
    render(<TinyGpsBackpropDiagram step={null} />);
    expect(screen.getByText(/Step through the exercise/)).toBeInTheDocument();
  });

  it("renders diagram when step exists", () => {
    render(<TinyGpsBackpropDiagram step={mockStep} />);
    expect(screen.getByText(/Backprop Flow/)).toBeInTheDocument();
    expect(screen.getByText("GRADIENTS")).toBeInTheDocument();
  });
});
