import { render, screen, waitFor } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { GdPage } from "./GdPage";

test("renders GD page with backend data", async () => {
  const fetchMock = vi.fn(async (url: RequestInfo) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.endsWith("/gd/loss-curves")) {
      return {
        ok: true,
        json: async () => ({
          points: [
            { p: 0.1, l1: 0.9, ce: 2.3026 },
            { p: 0.9, l1: 0.1, ce: 0.1053 },
          ],
        }),
      } as Response;
    }
    if (target.endsWith("/gd/token-losses")) {
      return {
        ok: true,
        json: async () => ({
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
    return { ok: false } as Response;
  }) as typeof fetch;
  global.fetch = fetchMock;

  render(<GdPage apiBase="http://127.0.0.1:8000" />);

  await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  expect(screen.getByText("Gradient Descent Exercises")).toBeInTheDocument();
  expect(screen.getByText("Token loss table")).toBeInTheDocument();
  expect(screen.getByText("Loss penalty vs p(correct)")).toBeInTheDocument();
});
