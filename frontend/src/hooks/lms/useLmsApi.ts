import { useCallback, useState } from "react";
import type { CustomConfig, LmsState, LmsStep } from "../../types";
import { buildCustomPayload } from "../../utils/custom";
import { useApi } from "../common/useApi";

type ResetOptions = {
  datasetName?: string;
  customConfig?: CustomConfig;
  customApplied?: boolean;
  lr?: number;
};

/**
 * Hook for LMS (Least Mean Squares) API operations.
 * Uses the centralized useApi hook for consistent error/loading handling.
 */
export function useLmsApi(apiBase: string) {
  const { error, setError, loading, get, post } = useApi(apiBase);
  const [state, setState] = useState<LmsState | null>(null);
  const [history, setHistory] = useState<LmsStep[]>([]);
  const [stepCount, setStepCount] = useState(0);

  const loadState = useCallback(async () => {
    const data = await get<LmsState>("/lms/state");
    if (data) {
      setState(data);
    }
  }, [get]);

  const step = useCallback(async () => {
    const data = await post<LmsStep>("/lms/step");
    if (data) {
      setHistory((prev) => [...prev, data].slice(-32));
      setStepCount((prev) => prev + 1);
      setState((prev) =>
        prev
          ? { ...prev, w: data.w_after, b: data.b_after, idx: data.idx, lr: data.lr, x: data.x, y: data.y }
          : {
              w: data.w_after,
              b: data.b_after,
              idx: data.idx,
              lr: data.lr,
              x: data.x,
              y: data.y,
              sample_count: 4,
              dataset: "or",
            },
      );
    }
  }, [post]);

  const resetWithOptions = useCallback(
    async (options: ResetOptions = {}) => {
      const body: Record<string, unknown> = {};
      if (options.datasetName) {
        if (options.datasetName === "custom") {
          if (!options.customApplied || !options.customConfig) {
            setError("Apply the custom dataset before loading LMS.");
            return;
          }
          Object.assign(body, { dataset: "custom" }, buildCustomPayload(options.customConfig));
        } else {
          body.dataset = options.datasetName;
        }
      }
      if (typeof options.lr === "number") {
        body.lr = options.lr;
      }
      const data = await post<LmsState>("/lms/reset", body);
      if (data) {
        setState(data);
        setHistory([]);
        setStepCount(0);
      }
    },
    [post, setError],
  );

  const reset = useCallback(async () => {
    await resetWithOptions();
  }, [resetWithOptions]);

  return { state, history, stepCount, error, loading, loadState, step, reset, resetWithOptions, setError };
}
