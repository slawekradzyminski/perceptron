import { useCallback, useState } from "react";
import type {
  DeepArchitecture,
  DeepBoundary,
  DeepComparisonEntry,
  DeepDatasetInfo,
  DeepHistoryEntry,
  DeepMetrics,
  DeepRegions,
  DeepResetOptions,
  DeepSample,
  DeepState,
  DeepStepInfo,
  DeepStepResponse,
} from "../../types/api";
import { useApi } from "../common/useApi";

// Re-export types for backward compatibility
export type {
  DeepArchitecture,
  DeepBoundary,
  DeepComparisonEntry,
  DeepDatasetInfo,
  DeepHistoryEntry,
  DeepMetrics,
  DeepRegions,
  DeepSample,
  DeepState,
  DeepStepInfo,
  DeepStepResponse,
};

type ResetOptions = DeepResetOptions;

/**
 * Hook for Deep Learning API operations.
 * Uses the centralized useApi hook for consistent error/loading handling.
 */
export function useDeepApi(apiBase: string) {
  const { error, setError, loading, get, post } = useApi(apiBase);
  const [state, setState] = useState<DeepState | null>(null);
  const [boundary, setBoundary] = useState<DeepBoundary | null>(null);
  const [regions, setRegions] = useState<DeepRegions | null>(null);
  const [history, setHistory] = useState<DeepHistoryEntry[]>([]);
  const [comparison, setComparison] = useState<DeepComparisonEntry[]>([]);

  const fetchState = useCallback(async () => {
    const data = await get<DeepState>("/deep/state");
    if (data) {
      setState(data);
    }
  }, [get]);

  const reset = useCallback(
    async (options: ResetOptions = {}) => {
      const data = await post<DeepState>("/deep/reset", options);
      if (data) {
        setState(data);
        setBoundary(null);
        setRegions(null);
        setHistory([]);
      }
    },
    [post],
  );

  const step = useCallback(
    async (batchSize: number = 1) => {
      const data = await post<DeepStepResponse>("/deep/step", { batch_size: batchSize });
      if (data) {
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
      }
    },
    [post],
  );

  const trainEpoch = useCallback(async () => {
    const data = await post<DeepStepResponse>("/deep/epoch");
    if (data) {
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
    }
  }, [post]);

  const fetchBoundary = useCallback(
    async (resolution: number = 50) => {
      const data = await get<DeepBoundary>(`/deep/boundary?resolution=${resolution}`);
      if (data) {
        setBoundary(data);
        setRegions({
          count: data.region_count,
          theoretical_max: data.theoretical_max,
          efficiency: data.region_count / data.theoretical_max,
        });
      }
    },
    [get],
  );

  const fetchRegions = useCallback(
    async (resolution: number = 50) => {
      const data = await get<DeepRegions>(`/deep/regions?resolution=${resolution}`);
      if (data) {
        setRegions(data);
      }
    },
    [get],
  );

  const addToComparison = useCallback(async () => {
    const entry = await post<DeepComparisonEntry>("/deep/comparison/add");
    if (entry) {
      setComparison((prev) => [...prev, entry]);
    }
  }, [post]);

  const clearComparison = useCallback(async () => {
    await post("/deep/comparison/clear");
    setComparison([]);
  }, [post]);

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
    setError,
  };
}
