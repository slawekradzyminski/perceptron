import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { TinyGpsHistoryTable } from "./TinyGpsHistoryTable";
import type { TinyGpsStep } from "../../../types";

const mockStep: TinyGpsStep = {
  dataset: "europe",
  mode: "1d",
  cities: ["paris"],
  idx: 0,
  sample_idx: 0,
  lr: 0.1,
  feature_order: ["lon"],
  sample: { x: [0.2], y: 0, city: "paris", lat: 0, lon: 0 },
  logits: [0.2],
  probs: [0.8],
  loss: 0.2,
  pred: 0,
  correct: true,
  grads: { m: [0.1], b: [0.2] },
  params_before: { m: [0.1], b: [0.2], M: [] },
  params_after: { m: [0.1], b: [0.2], M: [] },
  metrics_after: { loss: 0.2, accuracy: 1 },
  sample_count: 1,
};

describe("TinyGpsHistoryTable", () => {
  it("renders placeholder when empty", () => {
    render(<TinyGpsHistoryTable history={[]} />);
    expect(screen.getByText("No TinyGPS steps yet.")).toBeInTheDocument();
  });

  it("renders history rows", () => {
    render(<TinyGpsHistoryTable history={[mockStep]} />);
    expect(screen.getByText("paris")).toBeInTheDocument();
  });
});
