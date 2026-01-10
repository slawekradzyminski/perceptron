import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TransformerChatTab } from "./TransformerChatTab";
import type { OllamaStatus } from "../../hooks/transformer/useTransformerApi";

const mockStatus: OllamaStatus = {
  ok: true,
  model: "llama3.2:1b",
  base_url: "http://localhost:11434",
  models: ["llama3.2:1b"],
  model_available: true,
};

describe("TransformerChatTab", () => {
  it("sends chat messages", () => {
    const handleSend = vi.fn();
    render(
      <TransformerChatTab
        ollamaStatus={mockStatus}
        chatMessages={[]}
        streamingContent=""
        generating={false}
        error={null}
        onSend={handleSend}
        onReset={vi.fn()}
      />,
    );

    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "Hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(handleSend).toHaveBeenCalledWith("Hello");
  });
});
