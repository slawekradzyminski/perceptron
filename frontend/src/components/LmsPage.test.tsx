import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { LmsPage } from "./LmsPage";

test("renders LMS page and loads state", async () => {
  const fetchMock = vi.fn(async (url: RequestInfo, options?: RequestInit) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.endsWith("/lms/reset")) {
      return {
        ok: true,
        json: async () => ({
          w: [0, 0],
          b: 0,
          idx: 0,
          lr: 0.2,
          x: [-1, -1],
          y: -1,
          sample_count: 4,
          dataset: "or",
        }),
      } as Response;
    }
    if (target.endsWith("/lms/step") && options?.method === "POST") {
      return {
        ok: true,
        json: async () => ({
          x: [-1, -1],
          y: -1,
          w_before: [0, 0],
          b_before: 0,
          y_hat: 0,
          error: -1,
          grad_w1: 2,
          grad_w2: 2,
          grad_b: 2,
          w_after: [0.1, 0.1],
          b_after: -0.1,
          idx: 1,
          lr: 0.1,
        }),
      } as Response;
    }
    return {
      ok: true,
      json: async () => ({
        w: [0, 0],
        b: 0,
        idx: 0,
        lr: 0.1,
        x: [-1, -1],
        y: -1,
        sample_count: 4,
        dataset: "or",
      }),
    } as Response;
  }) as typeof fetch;
  global.fetch = fetchMock;

  render(
    <LmsPage
      apiBase="http://127.0.0.1:8000"
      datasetName="or"
      customConfig={{ rows: 1, cols: 2, samples: [] }}
      customApplied={false}
    />,
  );
  expect(screen.getByText("LMS (Widrow–Hoff) Exercise")).toBeInTheDocument();
  await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  const resetCall = fetchMock.mock.calls.find((call) =>
    typeof call[0] === "string" ? call[0].endsWith("/lms/reset") : call[0].url.endsWith("/lms/reset"),
  );
  expect(resetCall).toBeTruthy();
  const resetBody = JSON.parse(resetCall?.[1]?.body as string);
  expect(resetBody.lr).toBe(0.1);
  expect(screen.getByText("Click Step to start filling the LMS table.")).toBeInTheDocument();
  expect(screen.getByText("Error trend (E = (y − ŷ)²)")).toBeInTheDocument();

  fireEvent.keyDown(document, { key: "s" });
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
    expect.stringContaining("/lms/step"),
    expect.objectContaining({ method: "POST" }),
  ));
});
