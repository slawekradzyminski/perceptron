import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TinyGpsExerciseSetup } from "./TinyGpsExerciseSetup";
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
  params: { m: [0.1, 0.2], b: [0.3, 0.4], M: [] },
  next_sample: { x: [0.2], y: 0, city: "paris", lat: 0, lon: 0 },
  samples: [
    { x: [0.2], y: 0, city: "paris", lat: 0, lon: 0 },
    { x: [0.3], y: 1, city: "madrid", lat: 0, lon: 0 },
  ],
};

describe("TinyGpsExerciseSetup", () => {
  it("applies parsed values", () => {
    const handleApply = vi.fn();
    render(<TinyGpsExerciseSetup state={mockState} onApply={handleApply} />);

    const inputs = screen.getAllByRole("textbox");
    fireEvent.change(inputs[0], { target: { value: "1, 2" } });
    fireEvent.change(inputs[1], { target: { value: "3, 4" } });
    fireEvent.change(inputs[2], { target: { value: "1,0" } });

    fireEvent.click(screen.getByRole("button", { name: "Apply Exercise Params" }));

    expect(handleApply).toHaveBeenCalledWith(
      { m: [1, 2], b: [3, 4] },
      [1, 0],
    );
  });
});
