import { useCallback, useState } from "react";
import type {
  AttentionResult,
  KVCacheRequestConfig,
  KVCacheResult,
  LogitLensResult,
} from "../../types/api";
import { useApi } from "../common/useApi";

// Re-export types for backward compatibility
export type {
  AttentionResult,
  KVCacheResult,
  LogitLensResult,
} from "../../types/api";

export type {
  KVCacheArchitecture,
  KVCacheConfig,
  LogitLensLayer,
  LogitLensPrediction,
} from "../../types/api";

/**
 * Hook for Glass Box API operations (attention patterns, logit lens, KV cache).
 * Uses the centralized useApi hook for consistent error/loading handling.
 */
export function useGlassBoxApi(apiBase: string) {
  const { error, loading, post } = useApi(apiBase);
  const [attention, setAttention] = useState<AttentionResult | null>(null);
  const [logitLens, setLogitLens] = useState<LogitLensResult | null>(null);
  const [kvCache, setKvCache] = useState<KVCacheResult | null>(null);

  const fetchAttention = useCallback(
    async (text: string) => {
      const result = await post<AttentionResult>("/transformer/attention", { text });
      if (result) setAttention(result);
    },
    [post],
  );

  const fetchLogitLens = useCallback(
    async (text: string, topK: number = 5) => {
      const result = await post<LogitLensResult>("/transformer/logit-lens", { text, top_k: topK });
      if (result) setLogitLens(result);
    },
    [post],
  );

  const fetchKvCache = useCallback(
    async (config: KVCacheRequestConfig) => {
      const result = await post<KVCacheResult>("/transformer/kv-cache", config);
      if (result) setKvCache(result);
    },
    [post],
  );

  return {
    attention,
    logitLens,
    kvCache,
    error,
    loading,
    fetchAttention,
    fetchLogitLens,
    fetchKvCache,
  };
}
