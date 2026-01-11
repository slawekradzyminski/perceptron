import { useCallback, useEffect, useState } from "react";
import type {
  BackpropState,
  RegressionStep,
  TinyGpsStep,
} from "../../types";

type TinyGpsResetParams = {
  m?: number[];
  b?: number[];
  M?: number[][];
};

/**
 * NOTE: This hook does NOT use the centralized useApi hook because:
 *
 * 1. **Auto-loading on mount**: This hook auto-fetches state when mounted using useEffect.
 *    The useApi hook's setState calls within the effect trigger the ESLint
 *    react-hooks/set-state-in-effect rule when combined with auto-loading.
 *
 * 2. **Complex state updates**: The step functions update state by merging partial objects
 *    with the previous state, which requires access to the previous state value.
 *    This pattern doesn't map cleanly to the useApi return-and-set pattern.
 */
export function useBackpropApi(apiBase: string) {
  const [state, setState] = useState<BackpropState | null>(null);
  const [tinygpsHistory, setTinygpsHistory] = useState<TinyGpsStep[]>([]);
  const [regressionHistory, setRegressionHistory] = useState<RegressionStep[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loadState = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/backprop/state`);
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as BackpropState;
      setState(data);
    } catch {
      setError("API unreachable. Check backend.");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    void loadState();
  }, [loadState]);

  const resetTinygps = useCallback(
    async (dataset: string, lr: number, params?: TinyGpsResetParams, order?: number[]) => {
      setLoading(true);
      setError(null);
      try {
        const payload: Record<string, unknown> = { dataset, lr };
        if (params) payload.params = params;
        if (order) payload.order = order;
        const res = await fetch(`${apiBase}/backprop/tinygps/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const tinygps = (await res.json()) as BackpropState["tinygps"];
        setState((prev) => (prev ? { ...prev, tinygps } : null));
        setTinygpsHistory([]);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const stepTinygps = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/backprop/tinygps/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const step = (await res.json()) as TinyGpsStep;
      setTinygpsHistory((prev) => [...prev, step].slice(-32));
      setState((prev) =>
        prev
          ? {
              ...prev,
              tinygps: {
                ...prev.tinygps,
                idx: step.idx,
                params: step.params_after,
                lr: step.lr,
              },
            }
          : prev,
      );
    } catch {
      setError("API unreachable. Check backend.");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const resetRegression = useCallback(
    async (
      loss: "mse" | "l1",
      lr: number,
      samples?: { x: number; y: number }[],
      order?: number[],
      params?: { m?: number; b?: number },
    ) => {
      setLoading(true);
      setError(null);
      try {
        const payload: Record<string, unknown> = { loss, lr };
        if (samples) payload.samples = samples;
        if (order) payload.order = order;
        if (params) payload.params = params;
        const res = await fetch(`${apiBase}/backprop/regression/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const regression = (await res.json()) as BackpropState["regression"];
        setState((prev) => (prev ? { ...prev, regression } : null));
        setRegressionHistory([]);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const stepRegression = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/backprop/regression/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const step = (await res.json()) as RegressionStep;
      setRegressionHistory((prev) => [...prev, step].slice(-32));
      setState((prev) =>
        prev
          ? {
              ...prev,
              regression: {
                ...prev.regression,
                idx: step.idx,
                params: step.params_after,
                lr: step.lr,
              },
            }
          : prev,
      );
    } catch {
      setError("API unreachable. Check backend.");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  return {
    state,
    tinygpsHistory,
    regressionHistory,
    error,
    loading,
    loadState,
    resetTinygps,
    stepTinygps,
    resetRegression,
    stepRegression,
  };
}
