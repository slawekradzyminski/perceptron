import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { CustomModal } from "./CustomModal";

const baseProps = {
  isOpen: true,
  customConfig: { rows: 2, cols: 2, samples: [{ grid: [[-1, -1], [-1, -1]], y: -1 }] },
  onClose: vi.fn(),
  onRowsChange: vi.fn(),
  onColsChange: vi.fn(),
  onAddSample: vi.fn(),
  onApply: vi.fn(),
  onRemoveSample: vi.fn(),
  onToggleCell: vi.fn(),
  onLabelChange: vi.fn(),
};

test("renders modal and allows closing", () => {
  render(<CustomModal {...baseProps} />);
  expect(screen.getByRole("dialog")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  expect(baseProps.onClose).toHaveBeenCalled();
});

test("triggers apply", () => {
  render(<CustomModal {...baseProps} />);
  fireEvent.click(screen.getByRole("button", { name: "Apply custom dataset" }));
  expect(baseProps.onApply).toHaveBeenCalled();
});
