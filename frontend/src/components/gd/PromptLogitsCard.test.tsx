import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { PromptLogitsCard } from "./PromptLogitsCard";

const mockResponse = {
  prompt: "Hello",
  tokens: [
    { token: "world", prob: 0.2, logprob: -1.6, rank: 1 },
    { token: "there", prob: 0.1, logprob: -2.3, rank: 2 },
  ],
};

describe("PromptLogitsCard", () => {
  const apiBase = "http://localhost:8000";

  beforeEach(() => {
    vi.resetAllMocks();
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    }) as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("submits prompt and renders logits table", async () => {
    render(<PromptLogitsCard apiBase={apiBase} />);

    const input = screen.getByRole("textbox");
    fireEvent.change(input, { target: { value: "Hello" } });

    fireEvent.click(screen.getByRole("button", { name: "Fetch logits" }));

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/gd/next-token-logprobs"),
      );
    });

    expect(await screen.findByText("Probability")).toBeInTheDocument();
    expect(screen.getByText("world")).toBeInTheDocument();
  });
});
