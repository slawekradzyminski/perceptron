import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
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
  let call = 0;
  global.fetch = vi.fn(async () => {
    call += 1;
    if (call === 1) {
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
  }) as typeof fetch;

  render(<App />);
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
  global.fetch = vi.fn(async () => ({
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
  })) as typeof fetch;

  render(<App />);
  const select = await screen.findByLabelText("Dataset");
  fireEvent.change(select, { target: { value: "custom" } });
  expect(await screen.findByText("Custom dataset")).toBeInTheDocument();
});
