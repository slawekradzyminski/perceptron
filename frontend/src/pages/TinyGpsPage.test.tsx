import { render, screen, fireEvent } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { TinyGpsPage } from "./TinyGpsPage";

const mockState = {
  tinygps: {
    dataset: "madrid-paris-berlin",
    mode: "1d",
    cities: ["madrid", "paris", "berlin"],
    idx: 0,
    lr: 0.1,
    sample_count: 15,
    feature_order: ["lon"],
    sample_order: Array.from({ length: 15 }, (_, i) => i),
    params: { m: [0, 0, 0], b: [0, 0, 0], M: [] },
    next_sample: { x: [-3.7033], y: 0, city: "madrid", lat: 40.4167, lon: -3.7033 },
    samples: [
      { x: [-3.7033], y: 0, city: "madrid", lat: 40.4167, lon: -3.7033 },
      { x: [2.3514], y: 1, city: "paris", lat: 48.8575, lon: 2.3514 },
    ],
  },
  regression: {
    loss: "mse",
    idx: 0,
    lr: 0.1,
    sample_count: 4,
    params: { m: 0, b: 0 },
    next_sample: { x: 1, y: 3 },
    samples: [
      { x: 1, y: 3 },
      { x: 2, y: 5 },
      { x: 3, y: 7 },
      { x: 4, y: 9 },
    ],
    sample_order: [0, 1, 2, 3],
  },
  tinygps_datasets: ["paris-berlin", "paris-madrid", "madrid-paris-berlin", "four-cities"],
  tinygps_city_coords: {
    madrid: [[40.4167, -3.7033]],
    paris: [[48.8575, 2.3514]],
  },
};

const resetTinygps = vi.fn();
const stepTinygps = vi.fn();

vi.mock("../hooks/backprop/useBackpropApi", () => ({
  useBackpropApi: () => ({
    state: mockState,
    tinygpsHistory: [],
    regressionHistory: [],
    error: null,
    loading: false,
    loadState: vi.fn(),
    resetTinygps,
    stepTinygps,
    resetRegression: vi.fn(),
    stepRegression: vi.fn(),
  }),
}));

vi.mock("../hooks/common/useHotkeys", () => ({
  useHotkeys: vi.fn(),
}));

test("renders TinyGPS page and triggers step", async () => {
  render(<TinyGpsPage apiBase="http://127.0.0.1:8000" />);
  expect(screen.getByText("Backprop Lab (Chapter 3)")).toBeInTheDocument();
  expect(screen.getByText("TinyGPS Setup")).toBeInTheDocument();
  expect(screen.getByText("Book Table")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Step TinyGPS" }));
  expect(stepTinygps).toHaveBeenCalled();

  fireEvent.change(screen.getByLabelText("Dataset"), { target: { value: "paris-berlin" } });
  expect(resetTinygps).toHaveBeenCalled();

  const { useHotkeys } = await import("../hooks/common/useHotkeys");
  expect(useHotkeys).toHaveBeenCalled();
});
