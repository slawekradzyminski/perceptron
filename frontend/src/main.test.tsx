import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { MemoryRouter } from "react-router";
import App from "./App";

function mockCanvas() {
  HTMLCanvasElement.prototype.getContext = () =>
    ({
      canvas: { width: 100, height: 50 },
      clearRect: () => undefined,
      fillRect: () => undefined,
      beginPath: () => undefined,
      moveTo: () => undefined,
      lineTo: () => undefined,
      stroke: () => undefined,
      arc: () => undefined,
      fill: () => undefined,
      strokeRect: () => undefined,
      setLineDash: () => undefined,
      set fillStyle(_: string) {},
      set strokeStyle(_: string) {},
      set lineWidth(_: number) {},
    }) as unknown as CanvasRenderingContext2D;
}

test("renders and steps through backend", async () => {
  mockCanvas();
  global.fetch = vi.fn(async (url: RequestInfo, options?: RequestInit) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.endsWith("/state")) {
      return {
        ok: true,
        json: async () => ({
          w: [0, 0],
          b: 0,
          idx: 0,
          dataset: "or",
          next_x: [-1, -1],
          next_y: -1,
          grid_rows: 1,
          grid_cols: 2,
          sample_count: 4,
        }),
      } as Response;
    }
    if (target.endsWith("/error-surface")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "or",
          grid_rows: 1,
          grid_cols: 2,
          steps: 3,
          w_range: [-1, 1],
          bias: 0,
          sample_count: 4,
          grid: [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
          ],
        }),
      } as Response;
    }
    if (target.endsWith("/mlp-internals")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "or",
          grid_rows: 1,
          grid_cols: 2,
          hidden_dim: 2,
          sample_index: 0,
          sample_count: 4,
          x: [1, -1],
          y: 1,
          y01: 1,
          loss: 0.2,
          p_hat: 0.6,
          hidden: {
            weights_before: [[0.1, 0.2], [0.3, 0.4]],
            bias_before: [0, 0],
            weights_after: [[0.2, 0.3], [0.4, 0.5]],
            bias_after: [0.1, 0.1],
            z: [0.1, 0.2],
            a: [0.1, 0.2],
            templates_before: [[[0.1, 0.2]], [[0.3, 0.4]]],
            templates_after: [[[0.2, 0.3]], [[0.4, 0.5]]],
          },
          output: {
            weights_before: [[0.2, 0.3]],
            bias_before: [0],
            weights_after: [[0.3, 0.4]],
            bias_after: [0.1],
            z: 0.2,
            a: 0.6,
          },
          gradients: {
            hidden_W: [[0.1, 0.1], [0.2, 0.2]],
            hidden_b: [0.1, 0.1],
            out_W: [[0.1, 0.1]],
            out_b: [0.1],
            templates: [[[0.1, 0.1]], [[0.2, 0.2]]],
          },
        }),
      } as Response;
    }
    if (target.endsWith("/step") || target.endsWith("/reset")) {
      return {
        ok: true,
        json: async () => ({
          w: [1, 0],
          b: 1,
          idx: 1,
          dataset: "or",
          x: [1, -1],
          y: 1,
          score: 1,
          pred: 1,
          mistake: false,
          delta_w: [0, 0],
          delta_b: 0,
          lr: 1,
          next_x: [-1, 1],
          next_y: 1,
          grid_rows: 1,
          grid_cols: 2,
          sample_count: 4,
        }),
      } as Response;
    }
    return {
      ok: true,
      json: async () => ({
        w: [0, 0],
        b: 0,
        idx: 0,
        dataset: "or",
        next_x: [-1, -1],
        next_y: -1,
        grid_rows: 1,
        grid_cols: 2,
        sample_count: 4,
      }),
    } as Response;
  }) as typeof fetch;

  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>,
  );
  await waitFor(() => expect(screen.getByTestId("score-value")).toBeInTheDocument());

  const scoreValue = screen.getByTestId("score-value");
  expect(scoreValue.textContent).toBeDefined();

  fireEvent.click(screen.getByRole("button", { name: "Step" }));
  await waitFor(() => expect(screen.getByTestId("score-value").textContent).toBe("1.00"));

  fireEvent.click(screen.getByRole("button", { name: "Reset" }));
  await waitFor(() => expect((global.fetch as any).mock.calls.length).toBeGreaterThan(1));
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

test("shows custom dataset editor", async () => {
  mockCanvas();
  global.fetch = vi.fn(async (url: RequestInfo) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.endsWith("/error-surface")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "or",
          grid_rows: 1,
          grid_cols: 2,
          steps: 3,
          w_range: [-1, 1],
          bias: 0,
          sample_count: 4,
          grid: [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
          ],
        }),
      } as Response;
    }
    if (target.endsWith("/mlp-internals")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "or",
          grid_rows: 1,
          grid_cols: 2,
          hidden_dim: 2,
          sample_index: 0,
          sample_count: 4,
          x: [1, -1],
          y: 1,
          y01: 1,
          loss: 0.2,
          p_hat: 0.6,
          hidden: {
            weights_before: [[0.1, 0.2], [0.3, 0.4]],
            bias_before: [0, 0],
            weights_after: [[0.2, 0.3], [0.4, 0.5]],
            bias_after: [0.1, 0.1],
            z: [0.1, 0.2],
            a: [0.1, 0.2],
            templates_before: [[[0.1, 0.2]], [[0.3, 0.4]]],
            templates_after: [[[0.2, 0.3]], [[0.4, 0.5]]],
          },
          output: {
            weights_before: [[0.2, 0.3]],
            bias_before: [0],
            weights_after: [[0.3, 0.4]],
            bias_after: [0.1],
            z: 0.2,
            a: 0.6,
          },
          gradients: {
            hidden_W: [[0.1, 0.1], [0.2, 0.2]],
            hidden_b: [0.1, 0.1],
            out_W: [[0.1, 0.1]],
            out_b: [0.1],
            templates: [[[0.1, 0.1]], [[0.2, 0.2]]],
          },
        }),
      } as Response;
    }
    return {
      ok: true,
      json: async () => ({
        w: [0, 0],
        b: 0,
        idx: 0,
        dataset: "or",
        next_x: [-1, -1],
        next_y: -1,
        grid_rows: 1,
        grid_cols: 2,
        sample_count: 4,
      }),
    } as Response;
  }) as typeof fetch;

  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>,
  );
  const select = await screen.findByLabelText("Dataset");
  fireEvent.change(select, { target: { value: "custom" } });
  const dialog = await screen.findByRole("dialog");
  expect(dialog).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  expect(screen.queryByRole("dialog")).toBeNull();
});

test("switching from custom back to xor resets state", async () => {
  mockCanvas();
  const fetchMock = vi.fn(async (url: RequestInfo) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.endsWith("/error-surface")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "or",
          grid_rows: 1,
          grid_cols: 2,
          steps: 3,
          w_range: [-1, 1],
          bias: 0,
          sample_count: 4,
          grid: [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
          ],
        }),
      } as Response;
    }
    if (target.endsWith("/mlp-internals")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "or",
          grid_rows: 1,
          grid_cols: 2,
          hidden_dim: 2,
          sample_index: 0,
          sample_count: 4,
          x: [1, -1],
          y: 1,
          y01: 1,
          loss: 0.2,
          p_hat: 0.6,
          hidden: {
            weights_before: [[0.1, 0.2], [0.3, 0.4]],
            bias_before: [0, 0],
            weights_after: [[0.2, 0.3], [0.4, 0.5]],
            bias_after: [0.1, 0.1],
            z: [0.1, 0.2],
            a: [0.1, 0.2],
            templates_before: [[[0.1, 0.2]], [[0.3, 0.4]]],
            templates_after: [[[0.2, 0.3]], [[0.4, 0.5]]],
          },
          output: {
            weights_before: [[0.2, 0.3]],
            bias_before: [0],
            weights_after: [[0.3, 0.4]],
            bias_after: [0.1],
            z: 0.2,
            a: 0.6,
          },
          gradients: {
            hidden_W: [[0.1, 0.1], [0.2, 0.2]],
            hidden_b: [0.1, 0.1],
            out_W: [[0.1, 0.1]],
            out_b: [0.1],
            templates: [[[0.1, 0.1]], [[0.2, 0.2]]],
          },
        }),
      } as Response;
    }
    return {
      ok: true,
      json: async () => ({
        w: [0, 0],
        b: 0,
        idx: 0,
        dataset: "or",
        next_x: [-1, -1],
        next_y: -1,
        grid_rows: 1,
        grid_cols: 2,
        sample_count: 4,
      }),
    } as Response;
  }) as typeof fetch;
  global.fetch = fetchMock;

  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>,
  );
  const select = await screen.findByLabelText("Dataset");
  fireEvent.change(select, { target: { value: "custom" } });
  fireEvent.change(select, { target: { value: "xor" } });

  await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  const resetCall = [...fetchMock.mock.calls].reverse().find((call) => {
    const url = call[0] as string;
    return url.endsWith("/reset");
  });
  expect(resetCall).toBeTruthy();
  const body = JSON.parse(resetCall?.[1]?.body as string);
  expect(body.dataset).toBe("xor");
});

test("apply custom dataset posts custom payload", async () => {
  mockCanvas();
  const fetchMock = vi.fn(async (url: RequestInfo) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.endsWith("/error-surface")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "custom",
          grid_rows: 2,
          grid_cols: 2,
          steps: 3,
          w_range: [-1, 1],
          bias: 0,
          sample_count: 1,
          grid: [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
          ],
        }),
      } as Response;
    }
    if (target.endsWith("/mlp-internals")) {
      return {
        ok: true,
        json: async () => ({
          dataset: "custom",
          grid_rows: 2,
          grid_cols: 2,
          hidden_dim: 2,
          sample_index: 0,
          sample_count: 1,
          x: [-1, -1, -1, -1],
          y: -1,
          y01: 0,
          loss: 0.2,
          p_hat: 0.6,
          hidden: {
            weights_before: [[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6]],
            bias_before: [0, 0],
            weights_after: [[0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7]],
            bias_after: [0.1, 0.1],
            z: [0.1, 0.2],
            a: [0.1, 0.2],
            templates_before: [
              [
                [0.1, 0.2],
                [0.3, 0.4],
              ],
              [
                [0.3, 0.4],
                [0.5, 0.6],
              ],
            ],
            templates_after: [
              [
                [0.2, 0.3],
                [0.4, 0.5],
              ],
              [
                [0.4, 0.5],
                [0.6, 0.7],
              ],
            ],
          },
          output: {
            weights_before: [[0.2, 0.3]],
            bias_before: [0],
            weights_after: [[0.3, 0.4]],
            bias_after: [0.1],
            z: 0.2,
            a: 0.6,
          },
          gradients: {
            hidden_W: [
              [0.1, 0.1, 0.1, 0.1],
              [0.2, 0.2, 0.2, 0.2],
            ],
            hidden_b: [0.1, 0.1],
            out_W: [[0.1, 0.1]],
            out_b: [0.1],
            templates: [
              [
                [0.1, 0.1],
                [0.1, 0.1],
              ],
              [
                [0.2, 0.2],
                [0.2, 0.2],
              ],
            ],
          },
        }),
      } as Response;
    }
    return {
      ok: true,
      json: async () => ({
        w: [0, 0, 0, 0],
        b: 0,
        idx: 0,
        dataset: "custom",
        next_x: [-1, -1, -1, -1],
        next_y: -1,
        grid_rows: 2,
        grid_cols: 2,
        sample_count: 1,
      }),
    } as Response;
  }) as typeof fetch;
  global.fetch = fetchMock;

  render(
    <MemoryRouter>
      <App />
    </MemoryRouter>,
  );
  const select = await screen.findByLabelText("Dataset");
  fireEvent.change(select, { target: { value: "custom" } });

  const applyButton = await screen.findByRole("button", { name: "Apply custom dataset" });
  fireEvent.click(applyButton);

  await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  const lastCall = fetchMock.mock.calls.at(-1);
  const body = JSON.parse(lastCall?.[1]?.body as string);
  expect(body.dataset).toBe("custom");
  expect(body.grid_rows).toBe(2);
  expect(body.grid_cols).toBe(2);
  expect(body.samples?.length).toBe(1);
});
