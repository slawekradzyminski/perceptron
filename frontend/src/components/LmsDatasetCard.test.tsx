import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { LmsDatasetCard } from "./LmsDatasetCard";

const baseProps = {
  datasetName: "or",
  datasetDim: 2,
  datasetRows: [
    { x: [-1, -1], y: -1 },
    { x: [1, 1], y: 1 },
  ],
  canUseCustom: true,
  lmsLr: 0.1,
  onLrChange: vi.fn(),
  state: {
    w: [0, 0],
    b: 0,
    idx: 0,
    lr: 0.1,
    x: [-1, -1],
    y: -1,
    sample_count: 2,
    dataset: "or",
  },
  formatVec: (values: number[]) => `[${values.join(", ")}]`,
};

test("renders dataset card and handles lr changes", () => {
  const onLrChange = vi.fn();
  render(<LmsDatasetCard {...baseProps} onLrChange={onLrChange} />);
  expect(screen.getByText("Dataset")).toBeInTheDocument();
  expect(screen.getByText(/Active:/)).toBeInTheDocument();
  const input = screen.getByRole("spinbutton") as HTMLInputElement;
  fireEvent.change(input, { target: { value: "0.2" } });
  expect(onLrChange).toHaveBeenCalledWith(0.2);
});
