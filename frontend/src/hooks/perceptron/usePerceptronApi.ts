import { useCallback, useState } from "react";
import type { LastStep } from "../../types";
import type { CustomConfig } from "../../types";
import { buildCustomPayload } from "../../utils/custom";

/**
 * NOTE: This hook does NOT use the centralized useApi hook because:
 *
 * 1. **Complex internal state management**: This hook manages a single unified state object
 *    that combines API responses with local UI state (customApplied, lastStep, etc.)
 *
 * 2. **Config-based API calls**: Unlike other hooks that receive `apiBase` as a parameter,
 *    this hook receives the entire config object (with apiBase, lr, customConfig) on each call.
 *    This is needed because the perceptron page passes config per-call rather than at init time.
 *
 * 3. **Response-driven state updates**: The state update logic is deeply coupled with the
 *    response data, merging multiple fields conditionally. The useApi pattern of returning
 *    data and calling setState separately would require significant restructuring.
 *
 * Refactoring this hook would require changing the component interface significantly.
 */

export type ApiState = {
  datasetName: string;
  gridRows: number;
  gridCols: number;
  sampleCount: number;
  sampleIdx: number;
  w: number[];
  b: number;
  lastStep: LastStep | null;
  nextInput: { x: number[]; y: number } | null;
  apiError: string | null;
  customApplied: boolean;
};

export type ApiConfig = {
  apiBase: string;
  lr: number;
  customConfig: CustomConfig;
};

const DEFAULT_STATE: ApiState = {
  datasetName: "or",
  gridRows: 1,
  gridCols: 2,
  sampleCount: 4,
  sampleIdx: 0,
  w: [0, 0],
  b: 0,
  lastStep: null,
  nextInput: null,
  apiError: null,
  customApplied: false,
};

export function usePerceptronApi(initialState: ApiState = DEFAULT_STATE) {
  const [state, setState] = useState<ApiState>(initialState);

  const reset = useCallback(
    async (config: ApiConfig, datasetOverride?: string, forceCustomPayload = false) => {
      const dataset = datasetOverride ?? state.datasetName;
      try {
        const payload: Record<string, unknown> = {
          dataset,
          lr: config.lr,
        };
        if (dataset === "custom" && forceCustomPayload) {
          Object.assign(payload, buildCustomPayload(config.customConfig));
        }
        const res = await fetch(`${config.apiBase}/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          setState((prev) => ({ ...prev, apiError: `API error: ${res.status}` }));
          return;
        }
        const data = await res.json();
        setState((prev) => ({
          ...prev,
          datasetName: data.dataset ?? dataset,
          w: data.w ?? prev.w,
          b: data.b ?? prev.b,
          sampleIdx: data.idx ?? 0,
          gridRows: data.grid_rows ?? prev.gridRows,
          gridCols: data.grid_cols ?? prev.gridCols,
          sampleCount: data.sample_count ?? prev.sampleCount,
          nextInput: data.next_x && data.next_y ? { x: data.next_x, y: data.next_y } : null,
          lastStep: null,
          apiError: null,
          customApplied: data.dataset === "custom",
        }));
      } catch {
        setState((prev) => ({ ...prev, apiError: "API unreachable. Check backend." }));
      }
    },
    [state.datasetName],
  );

  const step = useCallback(
    async (config: ApiConfig) => {
      if (state.datasetName === "custom" && !state.customApplied) {
        setState((prev) => ({ ...prev, apiError: "Apply the custom dataset before stepping." }));
        return;
      }
      try {
        const res = await fetch(`${config.apiBase}/step`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset: state.datasetName, lr: config.lr }),
        });
        if (!res.ok) {
          setState((prev) => ({ ...prev, apiError: `API error: ${res.status}` }));
          return;
        }
        const data = await res.json();
        if (!Array.isArray(data?.w) || !Array.isArray(data?.x)) {
          setState((prev) => ({ ...prev, apiError: "API response missing step details. Restart backend." }));
          return;
        }
        setState((prev) => ({
          ...prev,
          datasetName: data.dataset ?? prev.datasetName,
          w: data.w,
          b: data.b,
          sampleIdx: data.idx ?? prev.sampleIdx,
          gridRows: data.grid_rows ?? prev.gridRows,
          gridCols: data.grid_cols ?? prev.gridCols,
          sampleCount: data.sample_count ?? prev.sampleCount,
          lastStep: {
            x: data.x,
            y: data.y,
            score: data.score,
            pred: data.pred,
            mistake: data.mistake,
            deltaW: data.delta_w,
            deltaB: data.delta_b,
            lr: data.lr ?? config.lr,
          },
          nextInput: data.next_x && data.next_y ? { x: data.next_x, y: data.next_y } : prev.nextInput,
          apiError: null,
        }));
      } catch {
        setState((prev) => ({ ...prev, apiError: "API unreachable. Check backend." }));
      }
    },
    [state.datasetName, state.customApplied],
  );

  return { state, setState, reset, step };
}
