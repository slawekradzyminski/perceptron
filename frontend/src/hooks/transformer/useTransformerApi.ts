import { useCallback, useState } from "react";

export type TransformerState = {
  current_text: string;
  current_token_count: number;
  encoding_name: string;
  vocab_size: number;
};

export type TokenItem = {
  id: number;
  text: string;
  position: number;
};

export type TokenizeResult = {
  text: string;
  token_count: number;
  tokens: TokenItem[];
};

export type TokenEmbedding = {
  id: number;
  text: string;
  position: number;
  embedding: number[];
  embedding_preview: number[];
  min: number;
  max: number;
  mean: number;
  std: number;
};

export type EmbedResult = {
  text: string;
  token_count: number;
  d_model: number;
  model_name?: string;
  matrix_shape: [number, number];
  explanation: string;
  tokens: TokenEmbedding[];
  full_matrix: number[][];
};

export type OperationDetails = {
  n_heads?: number;
  d_head?: number;
  q_shape?: [number, number];
  k_shape?: [number, number];
  v_shape?: [number, number];
  attention_scores_shape?: [number, number, number];
  hidden_dim?: number;
  input_shape?: [number, number];
  hidden_shape?: [number, number];
  output_shape?: [number, number];
  explanation: string;
};

export type BlockOperation = {
  name: string;
  formula: string;
  details: OperationDetails;
};

export type BlockInfo = {
  block: number;
  input_shape: [number, number];
  output_shape: [number, number];
  operations: BlockOperation[];
};

export type TraceResult = {
  text: string;
  token_count: number;
  d_model: number;
  n_heads: number;
  d_head: number;
  d_ff: number;
  n_blocks: number;
  blocks: BlockInfo[];
  final_shape: [number, number];
  explanation: string;
};

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatResult = {
  model_name?: string;
  response: string;
  generation_time_ms: number;
  error?: string;
};

export type OllamaStatus = {
  ok: boolean;
  model: string;
  base_url: string;
  models: string[];
  model_available: boolean;
  error?: string;
};

export type ModelInfo = {
  id: string;
  name: string;
  year: number;
  params: number;
  layers: number;
  heads: number;
  d_model: number;
  context_length: number;
};

export type ScaleModel = {
  name: string;
  year: number;
  params: number;
  params_formatted: string;
  type: string;
  description: string;
  input_size: string;
  notable: string;
};

export type ScaleComparison = {
  model1: ScaleModel;
  model2: ScaleModel;
  ratio: number;
  ratio_formatted: string;
  year_gap: number;
  explanation: string;
};

export type GrowthData = {
  data: {
    name: string;
    year: number;
    params: number;
    log_params: number;
    type: string;
  }[];
  insight: string;
};

export function useTransformerApi(apiBase: string) {
  const [state, setState] = useState<TransformerState | null>(null);
  const [tokens, setTokens] = useState<TokenizeResult | null>(null);
  const [embedInfo, setEmbedInfo] = useState<EmbedResult | null>(null);
  const [trace, setTrace] = useState<TraceResult | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [streamingContent, setStreamingContent] = useState<string>("");
  const [ollamaStatus, setOllamaStatus] = useState<OllamaStatus | null>(null);
  const [scaleModels, setScaleModels] = useState<ScaleModel[]>([]);
  const [comparison, setComparison] = useState<ScaleComparison | null>(null);
  const [growth, setGrowth] = useState<GrowthData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);

  const fetchState = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/transformer/state`);
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as TransformerState;
      setState(data);
      setError(null);
    } catch {
      setError("API unreachable. Check backend.");
    }
  }, [apiBase]);

  const tokenize = useCallback(
    async (text: string) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/transformer/tokenize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) {
          const errorData = await res.json();
          setError(`API error: ${errorData.detail || res.status}`);
          return;
        }
        const data = (await res.json()) as TokenizeResult;
        setTokens(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const getEmbedInfo = useCallback(
    async (text?: string) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/transformer/embed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(text ? { text } : {}),
        });
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const data = (await res.json()) as EmbedResult;
        setEmbedInfo(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const getTrace = useCallback(
    async (text?: string, nBlocks: number = 12) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/transformer/trace`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, n_blocks: nBlocks }),
        });
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const data = (await res.json()) as TraceResult;
        setTrace(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const fetchOllamaStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/transformer/ollama-status`);
      if (res.ok) {
        const data = (await res.json()) as OllamaStatus;
        setOllamaStatus(data);
      }
    } catch {
      setOllamaStatus({ ok: false, model: "", base_url: "", models: [], model_available: false });
    }
  }, [apiBase]);

  const sendChat = useCallback(
    async (prompt: string) => {
      setGenerating(true);
      setError(null);
      setStreamingContent("");

      // Add user message to history
      const userMessage: ChatMessage = { role: "user", content: prompt };
      const updatedMessages = [...chatMessages, userMessage];
      setChatMessages(updatedMessages);

      try {
        const res = await fetch(`${apiBase}/transformer/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: updatedMessages }),
        });

        if (!res.ok) {
          setError(`API error: ${res.status}`);
          setGenerating(false);
          return;
        }

        const reader = res.body?.getReader();
        if (!reader) {
          setError("Streaming not supported");
          setGenerating(false);
          return;
        }

        const decoder = new TextDecoder();
        let fullResponse = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value, { stream: true });
          const lines = text.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.token) {
                  fullResponse += data.token;
                  setStreamingContent(fullResponse);
                }
                if (data.done && data.error) {
                  setError(data.error);
                }
              } catch {
                // Ignore parse errors for incomplete chunks
              }
            }
          }
        }

        // Add assistant response to history
        if (fullResponse) {
          const assistantMessage: ChatMessage = { role: "assistant", content: fullResponse };
          setChatMessages([...updatedMessages, assistantMessage]);
        }
        setStreamingContent("");
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setGenerating(false);
      }
    },
    [apiBase, chatMessages],
  );

  const resetChat = useCallback(() => {
    setChatMessages([]);
    setStreamingContent("");
    setError(null);
  }, []);

  const fetchScaleModels = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/transformer/scale`);
      if (!res.ok) return;
      const data = (await res.json()) as { models: ScaleModel[] };
      setScaleModels(data.models);
    } catch {
      // Silently ignore
    }
  }, [apiBase]);

  const compareModels = useCallback(
    async (model1: string, model2: string) => {
      setLoading(true);
      try {
        const res = await fetch(
          `${apiBase}/transformer/scale/compare?model1=${encodeURIComponent(model1)}&model2=${encodeURIComponent(model2)}`,
        );
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const data = (await res.json()) as ScaleComparison;
        setComparison(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const fetchGrowth = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/transformer/scale/growth`);
      if (!res.ok) return;
      const data = (await res.json()) as GrowthData;
      setGrowth(data);
    } catch {
      // Silently ignore
    }
  }, [apiBase]);

  return {
    state,
    tokens,
    embedInfo,
    trace,
    chatMessages,
    streamingContent,
    ollamaStatus,
    scaleModels,
    comparison,
    growth,
    error,
    loading,
    generating,
    fetchState,
    tokenize,
    getEmbedInfo,
    getTrace,
    sendChat,
    resetChat,
    fetchOllamaStatus,
    fetchScaleModels,
    compareModels,
    fetchGrowth,
  };
}

