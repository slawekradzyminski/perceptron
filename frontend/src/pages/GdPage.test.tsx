import { render, screen, waitFor } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { GdPage } from "./GdPage";

test("renders GD page with backend data", async () => {
  const fetchMock = vi.fn(async (url: RequestInfo) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.startsWith("http://127.0.0.1:8000/gd/token-losses")) {
      return {
        ok: true,
        json: async () => ({
          source: "ollama",
          examples: [
            {
              id: "example",
              title: "Example sentence",
              description: "Example description.",
              average: { l1: 0.5, ce: 1.0 },
              rows: [
                {
                  context: "<begin_of_text>",
                  correct_token: "The",
                  p_correct: 0.5,
                  top_token: "The",
                  top_prob: 0.5,
                  l1_loss: 0.5,
                  ce_loss: 0.6931,
                },
              ],
            },
          ],
        }),
      } as Response;
    }
    if (target.endsWith("/gd/ollama-status")) {
      return {
        ok: true,
        json: async () => ({
          ok: true,
          model: "llama3.2:1b",
          base_url: "http://127.0.0.1:11434",
          last_latency_ms: 12.5,
          models: ["llama3.2:1b"],
        }),
      } as Response;
    }
    return { ok: false } as Response;
  }) as typeof fetch;
  global.fetch = fetchMock;

  render(<GdPage apiBase="http://127.0.0.1:8000" />);

  await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  expect(screen.getByText("Gradient Descent Exercises")).toBeInTheDocument();
  expect(screen.getByText("Ollama connected")).toBeInTheDocument();
  expect(screen.getByText("Token loss table")).toBeInTheDocument();
  expect(screen.getByText("Next-token logits")).toBeInTheDocument();
  expect(screen.getByText("Fetch logits")).toBeInTheDocument();
});
