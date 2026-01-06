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

  // Step and Reset buttons have been removed - keyboard shortcuts are used instead
  expect(screen.getByText(/Press/)).toBeInTheDocument();
  expect(screen.queryByLabelText("API base")).toBeNull();
});

test("shows customize button when enabled", () => {
  render(<Header {...baseProps} showCustomButton />);
  fireEvent.click(screen.getByRole("button", { name: "Customize" }));
  expect(baseProps.onOpenCustom).toHaveBeenCalled();
});

test("hides training controls on non-perceptron view", () => {
  render(<Header {...baseProps} showTrainingControls={false} route="lms" />);
  // No Step/Reset buttons anymore, but learning rate should be hidden
  expect(screen.queryByLabelText("Learning rate")).toBeNull();
});

test("shows LMS, MLP, and Backprop tabs", () => {
  render(<Header {...baseProps} />);
  expect(screen.getByRole("tab", { name: "Perceptron" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "LMS" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "MLP" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "GD" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "TinyGPS" })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "Regression" })).toBeInTheDocument();
});

test("hides dataset controls on GD view", () => {
  render(<Header {...baseProps} route="gd" showTrainingControls={false} />);
  expect(screen.queryByLabelText("Dataset")).toBeNull();
  expect(screen.queryByText("Switchboard")).toBeNull();
});

test("hides dataset controls on TinyGPS view", () => {
  render(<Header {...baseProps} route="tinygps" showTrainingControls={false} />);
  expect(screen.queryByLabelText("Dataset")).toBeNull();
  expect(screen.queryByText("Switchboard")).toBeNull();
});

test("hides dataset controls on Regression view", () => {
  render(<Header {...baseProps} route="regression" showTrainingControls={false} />);
  expect(screen.queryByLabelText("Dataset")).toBeNull();
  expect(screen.queryByText("Switchboard")).toBeNull();
});
