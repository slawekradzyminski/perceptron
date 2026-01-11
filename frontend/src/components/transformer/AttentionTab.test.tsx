import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { AttentionTab } from "./AttentionTab";
import type { AttentionResult } from "../../hooks/transformer/useGlassBoxApi";

const mockAttention: AttentionResult = {
  text: "Hello world",
  tokens: ["Hello", " world"],
  n_layers: 2,
  n_heads: 2,
  seq_len: 2,
  model_name: "gpt2",
  attentions: [
    [
      [[0.6, 0.4], [0.3, 0.7]],
      [[0.5, 0.5], [0.2, 0.8]],
    ],
    [
      [[0.7, 0.3], [0.4, 0.6]],
      [[0.8, 0.2], [0.1, 0.9]],
    ],
  ],
  explanation: "Attention patterns from gpt2",
};

describe("AttentionTab", () => {
  it("renders placeholder when no attention data", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <AttentionTab
        inputText="Hello world"
        onInputChange={handleInputChange}
        loading={false}
        attention={null}
        onFetchAttention={handleFetch}
      />,
    );

    expect(screen.getByRole("button", { name: "Analyze Attention" })).toBeInTheDocument();
    expect(
      screen.getByText(/Glass Box/),
    ).toBeInTheDocument();
  });

  it("triggers fetch on button click", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <AttentionTab
        inputText="Hello world"
        onInputChange={handleInputChange}
        loading={false}
        attention={null}
        onFetchAttention={handleFetch}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Analyze Attention" }));
    expect(handleFetch).toHaveBeenCalledWith("Hello world");
  });

  it("renders attention results with head selector", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <AttentionTab
        inputText="Hello world"
        onInputChange={handleInputChange}
        loading={false}
        attention={mockAttention}
        onFetchAttention={handleFetch}
      />,
    );

    expect(screen.getByText("gpt2")).toBeInTheDocument();
    expect(screen.getByText("2 layers × 2 heads")).toBeInTheDocument();
    expect(screen.getByText("2 tokens")).toBeInTheDocument();

    // Head selector should be visible (check labels exist)
    expect(screen.getByText("Layer")).toBeInTheDocument();
    expect(screen.getByText("Head")).toBeInTheDocument();
    // Check dropdowns exist
    expect(screen.getAllByRole("combobox").length).toBe(2);
  });

  it("shows educational content", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <AttentionTab
        inputText="Hello world"
        onInputChange={handleInputChange}
        loading={false}
        attention={mockAttention}
        onFetchAttention={handleFetch}
      />,
    );

    expect(screen.getByText("How to Read This")).toBeInTheDocument();
    expect(screen.getByText(/query positions/)).toBeInTheDocument();
  });

  it("disables button when loading", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <AttentionTab
        inputText="Hello world"
        onInputChange={handleInputChange}
        loading={true}
        attention={null}
        onFetchAttention={handleFetch}
      />,
    );

    const button = screen.getByRole("button", { name: "Analyzing..." });
    expect(button).toBeDisabled();
  });

  it("allows text editing via textarea", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <AttentionTab
        inputText="Hello world"
        onInputChange={handleInputChange}
        loading={false}
        attention={null}
        onFetchAttention={handleFetch}
      />,
    );

    const textarea = screen.getByRole("textbox");
    fireEvent.change(textarea, { target: { value: "New text" } });
    expect(handleInputChange).toHaveBeenCalledWith("New text");
  });
});
