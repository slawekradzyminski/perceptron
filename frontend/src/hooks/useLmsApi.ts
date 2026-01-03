import { useCallback, useState } from "react";
import type { LmsState, LmsStep } from "../types";

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
      setHistory((prev) => [data, ...prev].slice(0, 32));
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
            },
      );
    } catch {
      setError("API unreachable. Check backend.");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const reset = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/lms/reset`, { method: "POST", headers: { "Content-Type": "application/json" } });
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

  return { state, history, stepCount, error, loading, loadState, step, reset };
}
