import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TinyGpsStepSetupCard } from "./TinyGpsStepSetupCard";
import type { TinyGpsState } from "../../../types";

const mockState: TinyGpsState = {
  dataset: "europe",
  mode: "1d",
  cities: ["paris", "madrid"],
  idx: 0,
  lr: 0.1,
  sample_count: 2,
  feature_order: ["lon"],
  sample_order: [0, 1],
  params: { m: [0.1, -0.2], b: [0.3, -0.1], M: [] },
  next_sample: { x: [0.2], y: 0, city: "paris", lat: 0, lon: 0 },
  samples: [
    { x: [0.2], y: 0, city: "paris", lat: 0, lon: 0 },
    { x: [0.3], y: 1, city: "madrid", lat: 0, lon: 0 },
  ],
};

describe("TinyGpsStepSetupCard", () => {
  it("renders current params and applies setup", () => {
    const handleApply = vi.fn();
    render(<TinyGpsStepSetupCard state={mockState} onApply={handleApply} />);

    expect(screen.getByText(/Current Params/)).toBeInTheDocument();

    const applyButton = screen.getByRole("button", { name: "Apply Exercise Params" });
    fireEvent.click(applyButton);

    expect(handleApply).toHaveBeenCalled();
  });
});
