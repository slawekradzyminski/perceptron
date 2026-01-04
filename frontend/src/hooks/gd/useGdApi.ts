import { useEffect, useState } from "react";
import type { TokenLossResponse, OllamaStatusResponse } from "../../types";

export function useGdApi(apiBase: string) {
  const [tokenLosses, setTokenLosses] = useState<TokenLossResponse | null>(null);
  const [status, setStatus] = useState<OllamaStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const tokenRes = await fetch(`${apiBase}/gd/token-losses?source=ollama`);
        if (!tokenRes.ok) throw new Error("Failed to load token losses");
        const tokenData = (await tokenRes.json()) as TokenLossResponse;

        const statusRes = await fetch(`${apiBase}/gd/ollama-status`);
        const statusData = statusRes.ok ? ((await statusRes.json()) as OllamaStatusResponse) : null;

        if (mounted) {
          setTokenLosses(tokenData);
          setStatus(statusData);
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err.message : "Failed to load GD data");
        }
      } finally {
        if (mounted) setLoading(false);
      }
    };
    void load();
    return () => {
      mounted = false;
    };
  }, [apiBase]);

  return { tokenLosses, status, error, loading };
}
