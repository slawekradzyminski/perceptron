import { act, renderHook } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { useLmsApi } from "./useLmsApi";

test("loads state and steps LMS", async () => {
  const fetchMock = vi.fn(async (url: RequestInfo) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.endsWith("/lms/state")) {
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
        }),
      } as Response;
    }
    if (target.endsWith("/lms/step")) {
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
    if (target.endsWith("/lms/reset")) {
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
      }),
    } as Response;
  }) as typeof fetch;
  global.fetch = fetchMock;

  const { result } = renderHook(() => useLmsApi("http://127.0.0.1:8000"));

  await act(async () => {
    await result.current.loadState();
  });
  expect(result.current.state?.w).toEqual([0, 0]);

  await act(async () => {
    await result.current.step();
  });
  expect(result.current.history.length).toBe(1);
  expect(result.current.state?.w).toEqual([0.1, 0.1]);
});
