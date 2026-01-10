import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TransformerTraceTab } from "./TransformerTraceTab";
import type { TraceResult } from "../../hooks/transformer/useTransformerApi";

const mockTrace: TraceResult = {
  text: "Hello",
  token_count: 2,
  d_model: 4,
  n_heads: 2,
  d_head: 2,
  d_ff: 8,
  n_blocks: 1,
  blocks: [
    {
      block: 1,
      input_shape: [2, 4],
      output_shape: [2, 4],
      operations: [
        {
          name: "Attention",
          formula: "QK^T",
          details: { explanation: "Attention detail", n_heads: 2, d_head: 2 },
        },
      ],
    },
  ],
  final_shape: [2, 4],
  explanation: "Trace explanation",
};

describe("TransformerTraceTab", () => {
  it("renders trace data and handles trace requests", () => {
    const handleTrace = vi.fn();
    render(
      <TransformerTraceTab
        inputText="Hello"
        loading={false}
        trace={mockTrace}
        onTrace={handleTrace}
      />,
    );

    expect(screen.getByText("Block 1")).toBeInTheDocument();

    const input = screen.getByRole("spinbutton");
    fireEvent.change(input, { target: { value: "5" } });
    fireEvent.click(screen.getByRole("button", { name: "Show Trace" }));

    expect(handleTrace).toHaveBeenCalledWith(5);
  });
});
