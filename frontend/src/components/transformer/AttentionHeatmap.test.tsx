import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { AttentionHeatmap } from "./AttentionHeatmap";

const mockMatrix = [
  [0.6, 0.4],
  [0.3, 0.7],
];

const mockTokens = ["Hello", " world"];

describe("AttentionHeatmap", () => {
  it("renders heatmap grid", () => {
    render(<AttentionHeatmap matrix={mockMatrix} tokens={mockTokens} />);

    // Should have cells for the 2x2 matrix
    const cells = document.querySelectorAll(".heatmap-cell");
    expect(cells.length).toBe(4);
  });

  it("renders token labels", () => {
    render(<AttentionHeatmap matrix={mockMatrix} tokens={mockTokens} />);

    // Row and column labels should be present
    expect(screen.getAllByText(/Hello/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/world/i).length).toBeGreaterThan(0);
  });

  it("renders legend", () => {
    render(<AttentionHeatmap matrix={mockMatrix} tokens={mockTokens} />);

    expect(screen.getByText("Low attention")).toBeInTheDocument();
    expect(screen.getByText("High attention")).toBeInTheDocument();
  });

  it("shows tooltip on hover", () => {
    render(<AttentionHeatmap matrix={mockMatrix} tokens={mockTokens} />);

    const cells = document.querySelectorAll(".heatmap-cell");
    fireEvent.mouseEnter(cells[0]);

    // Tooltip should show the attention weight
    expect(screen.getByText("0.6000")).toBeInTheDocument();
  });

  it("hides tooltip on mouse leave", () => {
    render(<AttentionHeatmap matrix={mockMatrix} tokens={mockTokens} />);

    const cells = document.querySelectorAll(".heatmap-cell");
    fireEvent.mouseEnter(cells[0]);

    expect(screen.getByText("0.6000")).toBeInTheDocument();

    fireEvent.mouseLeave(cells[0]);

    expect(screen.queryByText("0.6000")).not.toBeInTheDocument();
  });

  it("applies colors based on attention weight", () => {
    render(<AttentionHeatmap matrix={mockMatrix} tokens={mockTokens} />);

    const cells = document.querySelectorAll(".heatmap-cell");

    // Higher attention (0.7) should have different color than lower (0.3)
    const cell1 = cells[1] as HTMLElement; // 0.4
    const cell2 = cells[3] as HTMLElement; // 0.7

    expect(cell1.style.backgroundColor).not.toBe("");
    expect(cell2.style.backgroundColor).not.toBe("");
    // Colors should be different (higher attention = darker)
    expect(cell1.style.backgroundColor).not.toBe(cell2.style.backgroundColor);
  });

  it("handles larger matrices", () => {
    const largeMatrix = [
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
    ];
    const largeTokens = ["A", "B", "C", "D"];

    render(<AttentionHeatmap matrix={largeMatrix} tokens={largeTokens} />);

    const cells = document.querySelectorAll(".heatmap-cell");
    expect(cells.length).toBe(16);
  });

  it("truncates long token text", () => {
    const longTokens = ["VeryLongTokenThatShouldBeTruncated", "Short"];
    const matrix = [
      [0.5, 0.5],
      [0.5, 0.5],
    ];

    render(<AttentionHeatmap matrix={matrix} tokens={longTokens} />);

    // Should not show the full long token (it gets truncated)
    expect(
      screen.queryByText("VeryLongTokenThatShouldBeTruncated"),
    ).not.toBeInTheDocument();
  });
});
