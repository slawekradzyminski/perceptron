import { render, screen, fireEvent } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { Header } from "./Header";

const baseProps = {
  datasetName: "or",
  gridRows: 1,
  gridCols: 2,
  lr: 1,
  showCustomButton: false,
  route: "main" as const,
  showTrainingControls: true,
  onDatasetChange: vi.fn(),
  onStep: vi.fn(),
  onReset: vi.fn(),
  onLrChange: vi.fn(),
  onOpenCustom: vi.fn(),
  onRouteChange: vi.fn(),
};

test("renders header controls and triggers actions", () => {
  render(<Header {...baseProps} />);
  fireEvent.change(screen.getByLabelText("Dataset"), { target: { value: "xor" } });
  expect(baseProps.onDatasetChange).toHaveBeenCalledWith("xor");

  fireEvent.click(screen.getByRole("button", { name: "Step" }));
  expect(baseProps.onStep).toHaveBeenCalled();

  fireEvent.click(screen.getByRole("button", { name: "Reset" }));
  expect(baseProps.onReset).toHaveBeenCalled();

  expect(screen.queryByLabelText("API base")).toBeNull();
});

test("shows customize button when enabled", () => {
  render(<Header {...baseProps} showCustomButton />);
  fireEvent.click(screen.getByRole("button", { name: "Customize" }));
  expect(baseProps.onOpenCustom).toHaveBeenCalled();
});

test("hides training controls on diagnostics view", () => {
  render(<Header {...baseProps} showTrainingControls={false} route="diagnostics" />);
  expect(screen.queryByRole("button", { name: "Step" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Reset" })).toBeNull();
});
