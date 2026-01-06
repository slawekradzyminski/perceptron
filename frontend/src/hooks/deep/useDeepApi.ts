import { useCallback, useState } from "react";

export type DeepArchitecture = {
  input_dim: number;
  hidden_dims: number[];
  output_dim: number;
  depth: number;
  width: number;
  param_count: number;
};

export type DeepMetrics = {
  loss: number;
  accuracy: number;
};

export type DeepDatasetInfo = {
  name: string;
  description: string;
  n_classes: number;
};

export type DeepSample = {
  x: number[];
  y: number;
};

export type DeepState = {
  dataset: string;
  dataset_info: DeepDatasetInfo;
  sample_count: number;
  idx: number;
  epoch: number;
  total_steps: number;
  lr: number;
  architecture: DeepArchitecture;
  metrics: DeepMetrics;
  samples: DeepSample[];
};

export type DeepStepInfo = {
  batch_size: number;
  last_loss: number;
  last_correct: boolean;
  last_prediction: number;
  grad_norm: number;
};

export type DeepStepResponse = DeepState & {
  step_info: DeepStepInfo;
};

export type DeepBoundary = {
  resolution: number;
  predictions: number[][];
  region_ids: number[][];
  region_count: number;
  theoretical_max: number;
};

export type DeepRegions = {
  count: number;
  theoretical_max: number;
  efficiency: number;
};

export type DeepHistoryEntry = {
  step: number;
  epoch: number;
  loss: number;
  accuracy: number;
};

export type DeepComparisonEntry = {
  depth: number;
  width: number;
  hidden_dims: number[];
  param_count: number;
  actual_regions: number;
  theoretical_max: number;
  accuracy: number;
  loss: number;
  total_steps: number;
};

type ResetOptions = {
  dataset?: string;
  hidden_dims?: number[];
  lr?: number;
  seed?: number;
};

export function useDeepApi(apiBase: string) {
  const [state, setState] = useState<DeepState | null>(null);
  const [boundary, setBoundary] = useState<DeepBoundary | null>(null);
  const [regions, setRegions] = useState<DeepRegions | null>(null);
  const [history, setHistory] = useState<DeepHistoryEntry[]>([]);
  const [comparison, setComparison] = useState<DeepComparisonEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchState = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/deep/state`);
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as DeepState;
      setState(data);
      setError(null);
    } catch {
      setError("API unreachable. Check backend.");
    }
  }, [apiBase]);

  const reset = useCallback(
    async (options: ResetOptions = {}) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/deep/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(options),
        });
        if (!res.ok) {
          const text = await res.text();
          setError(`API error: ${res.status} - ${text}`);
          return;
        }
        const data = (await res.json()) as DeepState;
        setState(data);
        setBoundary(null);
        setRegions(null);
        setHistory([]);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const step = useCallback(
    async (batchSize: number = 1) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/deep/step`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ batch_size: batchSize }),
        });
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const data = (await res.json()) as DeepStepResponse;
        setState(data);
        // Add to local history
        setHistory((prev) => [
          ...prev,
          {
            step: data.total_steps,
            epoch: data.epoch,
            loss: data.metrics.loss,
            accuracy: data.metrics.accuracy,
          },
        ]);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const trainEpoch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/deep/epoch`, {
        method: "POST",
      });
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as DeepStepResponse;
      setState(data);
      setHistory((prev) => [
        ...prev,
        {
          step: data.total_steps,
          epoch: data.epoch,
          loss: data.metrics.loss,
          accuracy: data.metrics.accuracy,
        },
      ]);
    } catch {
      setError("API unreachable. Check backend.");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const fetchBoundary = useCallback(
    async (resolution: number = 50) => {
      setLoading(true);
      try {
        const res = await fetch(`${apiBase}/deep/boundary?resolution=${resolution}`);
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const data = (await res.json()) as DeepBoundary;
        setBoundary(data);
        setRegions({
          count: data.region_count,
          theoretical_max: data.theoretical_max,
          efficiency: data.region_count / data.theoretical_max,
        });
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const fetchRegions = useCallback(
    async (resolution: number = 50) => {
      try {
        const res = await fetch(`${apiBase}/deep/regions?resolution=${resolution}`);
        if (!res.ok) return;
        const data = (await res.json()) as DeepRegions;
        setRegions(data);
      } catch {
        // Silently ignore
      }
    },
    [apiBase],
  );

  const addToComparison = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/deep/comparison/add`, {
        method: "POST",
      });
      if (!res.ok) return;
      const entry = (await res.json()) as DeepComparisonEntry;
      setComparison((prev) => [...prev, entry]);
    } catch {
      // Silently ignore
    }
  }, [apiBase]);

  const clearComparison = useCallback(async () => {
    try {
      await fetch(`${apiBase}/deep/comparison/clear`, { method: "POST" });
      setComparison([]);
    } catch {
      // Silently ignore
    }
  }, [apiBase]);

  return {
    state,
    boundary,
    regions,
    history,
    comparison,
    error,
    loading,
    fetchState,
    reset,
    step,
    trainEpoch,
    fetchBoundary,
    fetchRegions,
    addToComparison,
    clearComparison,
  };
}

