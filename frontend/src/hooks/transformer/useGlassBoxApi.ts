import { useCallback, useState } from "react";

export type AttentionResult = {
  text: string;
  tokens: string[];
  n_layers: number;
  n_heads: number;
  seq_len: number;
  model_name: string;
  attentions: number[][][][]; // [layers][heads][seq][seq]
  explanation: string;
};

export type LogitLensPrediction = {
  token: string;
  token_id: number;
  probability: number;
};

export type LogitLensLayer = {
  layer: number;
  layer_name: string;
  predictions: LogitLensPrediction[];
};

export type LogitLensResult = {
  text: string;
  tokens: string[];
  n_layers: number;
  top_k: number;
  model_name: string;
  layers: LogitLensLayer[];
  explanation: string;
};

export type KVCacheArchitecture = {
  name: string;
  full_name: string;
  description: string;
  memory_bytes: number;
  memory_formatted: string;
  ratio_to_mha: number;
  kv_heads?: number;
  latent_dim?: number;
};

export type KVCacheConfig = {
  context_length: number;
  n_layers: number;
  d_model: number;
  n_heads: number;
  d_head: number;
  gqa_groups: number;
  mla_latent_dim: number;
  bytes_per_param: number;
};

export type KVCacheResult = {
  config: KVCacheConfig;
  architectures: KVCacheArchitecture[];
  mha_to_mla_savings: number;
  explanation: string;
};

export function useGlassBoxApi(apiBase: string) {
  const [attention, setAttention] = useState<AttentionResult | null>(null);
  const [logitLens, setLogitLens] = useState<LogitLensResult | null>(null);
  const [kvCache, setKvCache] = useState<KVCacheResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchAttention = useCallback(
    async (text: string) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/transformer/attention`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) {
          const errorData = await res.json();
          setError(`API error: ${errorData.detail || res.status}`);
          return;
        }
        const data = (await res.json()) as AttentionResult;
        setAttention(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const fetchLogitLens = useCallback(
    async (text: string, topK: number = 5) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/transformer/logit-lens`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, top_k: topK }),
        });
        if (!res.ok) {
          const errorData = await res.json();
          setError(`API error: ${errorData.detail || res.status}`);
          return;
        }
        const data = (await res.json()) as LogitLensResult;
        setLogitLens(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const fetchKvCache = useCallback(
    async (config: {
      context_length?: number;
      n_layers?: number;
      d_model?: number;
      n_heads?: number;
      gqa_groups?: number;
      mla_latent_dim?: number;
    }) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/transformer/kv-cache`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config),
        });
        if (!res.ok) {
          const errorData = await res.json();
          setError(`API error: ${errorData.detail || res.status}`);
          return;
        }
        const data = (await res.json()) as KVCacheResult;
        setKvCache(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
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
