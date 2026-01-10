import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TransformerTokenizeTab } from "./TransformerTokenizeTab";
import type { EmbedResult, TokenizeResult } from "../../hooks/transformer/useTransformerApi";

const mockTokens: TokenizeResult = {
  text: "Hello",
  token_count: 1,
  tokens: [{ id: 1, text: "Hello", position: 0 }],
};

const mockEmbed: EmbedResult = {
  text: "Hello",
  token_count: 1,
  d_model: 2,
  matrix_shape: [1, 2],
  explanation: "Embedding info",
  tokens: [
    {
      id: 1,
      text: "Hello",
      position: 0,
      embedding: [0.1, 0.2],
      embedding_preview: [0.1, 0.2],
      min: 0.1,
      max: 0.2,
      mean: 0.15,
      std: 0.05,
    },
  ],
  full_matrix: [[0.1, 0.2]],
};

describe("TransformerTokenizeTab", () => {
  it("renders tokens and triggers tokenize", () => {
    const handleTokenize = vi.fn();
    render(
      <TransformerTokenizeTab
        inputText="Hello"
        loading={false}
        tokens={mockTokens}
        embedInfo={mockEmbed}
        onTokenize={handleTokenize}
      />,
    );

    expect(screen.getByText("Tokens (1)")).toBeInTheDocument();
    expect(screen.getByText("Embedding Matrix")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Tokenize & Embed" }));
    expect(handleTokenize).toHaveBeenCalled();
  });
});
