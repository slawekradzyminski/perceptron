import { render, screen, fireEvent } from "@testing-library/react";
import { expect, test, vi, describe } from "vitest";
import { DeepComparisonTable } from "./DeepComparisonTable";

const mockEntries = [
  {
    depth: 1,
    width: 8,
    hidden_dims: [8],
    param_count: 42,
    actual_regions: 5,
    theoretical_max: 37,
    accuracy: 0.75,
    loss: 0.5,
    total_steps: 100,
  },
  {
    depth: 2,
    width: 16,
    hidden_dims: [16, 16],
    param_count: 370,
    actual_regions: 25,
    theoretical_max: 145,
    accuracy: 0.85,
    loss: 0.3,
    total_steps: 200,
  },
];

describe("DeepComparisonTable", () => {
  test("renders empty state with hint", () => {
    render(<DeepComparisonTable entries={[]} onClear={vi.fn()} />);

    expect(screen.getByText(/Architecture Comparison/)).toBeInTheDocument();
    expect(screen.getByText(/Exercise 4.9/)).toBeInTheDocument();
    expect(screen.getByText(/Train different architectures/)).toBeInTheDocument();
  });

  test("renders table headers when entries exist", () => {
    render(<DeepComparisonTable entries={mockEntries} onClear={vi.fn()} />);

    expect(screen.getByText("Layers")).toBeInTheDocument();
    expect(screen.getByText("Width")).toBeInTheDocument();
    expect(screen.getByText("Params")).toBeInTheDocument();
    expect(screen.getByText("Regions")).toBeInTheDocument();
    expect(screen.getByText("Theory")).toBeInTheDocument();
    expect(screen.getByText("Accuracy")).toBeInTheDocument();
    expect(screen.getByText("Steps")).toBeInTheDocument();
  });

  test("renders entry data correctly", () => {
    render(<DeepComparisonTable entries={mockEntries} onClear={vi.fn()} />);

    // First entry
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("37")).toBeInTheDocument();
    expect(screen.getByText("75.0%")).toBeInTheDocument();

    // Second entry
    expect(screen.getByText("370")).toBeInTheDocument();
    expect(screen.getByText("145")).toBeInTheDocument();
    expect(screen.getByText("85.0%")).toBeInTheDocument();
  });

  test("renders correct number of rows", () => {
    render(<DeepComparisonTable entries={mockEntries} onClear={vi.fn()} />);

    const rows = screen.getAllByRole("row");
    // 1 header row + 2 data rows
    expect(rows).toHaveLength(3);
  });

  test("renders Clear button when entries exist", () => {
    render(<DeepComparisonTable entries={mockEntries} onClear={vi.fn()} />);

    expect(screen.getByText("Clear")).toBeInTheDocument();
  });

  test("calls onClear when Clear button is clicked", () => {
    const onClear = vi.fn();
    render(<DeepComparisonTable entries={mockEntries} onClear={onClear} />);

    const clearButton = screen.getByText("Clear");
    fireEvent.click(clearButton);

    expect(onClear).toHaveBeenCalledTimes(1);
  });

  test("does not show Clear button when empty", () => {
    render(<DeepComparisonTable entries={[]} onClear={vi.fn()} />);

    expect(screen.queryByText("Clear")).not.toBeInTheDocument();
  });

  test("renders single entry correctly", () => {
    render(<DeepComparisonTable entries={[mockEntries[0]]} onClear={vi.fn()} />);

    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(2); // 1 header + 1 data row
  });

  test("displays depth values", () => {
    render(<DeepComparisonTable entries={mockEntries} onClear={vi.fn()} />);

    // depth values in cells
    const cells = screen.getAllByRole("cell");
    // First row first cell is depth
    expect(cells[0]).toHaveTextContent("1");
    // Second row first cell
    expect(cells[7]).toHaveTextContent("2");
  });

  test("displays width values", () => {
    render(<DeepComparisonTable entries={mockEntries} onClear={vi.fn()} />);

    // 8 for first entry, 16 for second
    expect(screen.getByText("8")).toBeInTheDocument();
    expect(screen.getByText("16")).toBeInTheDocument();
  });

  test("displays steps values", () => {
    render(<DeepComparisonTable entries={mockEntries} onClear={vi.fn()} />);

    expect(screen.getByText("100")).toBeInTheDocument();
    expect(screen.getByText("200")).toBeInTheDocument();
  });

  test("highlights regions column", () => {
    render(<DeepComparisonTable entries={mockEntries} onClear={vi.fn()} />);

    // Check that Regions column data has highlight class
    const cells = screen.getAllByRole("cell");
    const regionsCells = cells.filter(
      (cell) => cell.classList.contains("highlight"),
    );
    expect(regionsCells.length).toBe(2); // One for each entry
  });
});

