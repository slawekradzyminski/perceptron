import { render, screen, fireEvent } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { Header } from "../components/Header";

const baseProps = {
  datasetName: "or",
  gridRows: 1,
  gridCols: 2,
  lr: 1,
  apiBase: "http://127.0.0.1:8000",
  showCustomButton: false,
  onDatasetChange: vi.fn(),
  onStep: vi.fn(),
  onReset: vi.fn(),
  onLrChange: vi.fn(),
  onApiBaseChange: vi.fn(),
  onOpenCustom: vi.fn(),
};

test("renders header controls and triggers actions", () => {
  render(<Header {...baseProps} />);
  fireEvent.change(screen.getByLabelText("Dataset"), { target: { value: "xor" } });
  expect(baseProps.onDatasetChange).toHaveBeenCalledWith("xor");

  fireEvent.click(screen.getByRole("button", { name: "Step" }));
  expect(baseProps.onStep).toHaveBeenCalled();

  fireEvent.click(screen.getByRole("button", { name: "Reset" }));
  expect(baseProps.onReset).toHaveBeenCalled();
});

test("shows customize button when enabled", () => {
  render(<Header {...baseProps} showCustomButton />);
  fireEvent.click(screen.getByRole("button", { name: "Customize" }));
  expect(baseProps.onOpenCustom).toHaveBeenCalled();
});
