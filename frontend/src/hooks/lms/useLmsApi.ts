import { useCallback, useState } from "react";
import type { CustomConfig, LmsState, LmsStep } from "../types";
import { buildCustomPayload } from "../../utils/custom";

type ResetOptions = {
  datasetName?: string;
  customConfig?: CustomConfig;
  customApplied?: boolean;
  lr?: number;
};

export function useLmsApi(apiBase: string) {
  const [state, setState] = useState<LmsState | null>(null);
  const [history, setHistory] = useState<LmsStep[]>([]);
  const [stepCount, setStepCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadState = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/lms/state`);
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as LmsState;
      setState(data);
    } catch {
      setError("API unreachable. Check backend.");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const step = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/lms/step`, { method: "POST", headers: { "Content-Type": "application/json" } });
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as LmsStep;
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
    } catch {
      setError("API unreachable. Check backend.");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const resetWithOptions = useCallback(async (options: ResetOptions = {}) => {
    setLoading(true);
    setError(null);
    try {
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
      const res = await fetch(`${apiBase}/lms/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as LmsState;
      setState(data);
      setHistory([]);
      setStepCount(0);
    } catch {
      setError("API unreachable. Check backend.");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const reset = useCallback(async () => {
    await resetWithOptions();
  }, [resetWithOptions]);

  return { state, history, stepCount, error, loading, loadState, step, reset, resetWithOptions };
}
