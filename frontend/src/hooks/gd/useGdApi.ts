import { useEffect, useState } from "react";
import type { TokenLossResponse, OllamaStatusResponse } from "../../types";

/**
 * NOTE: This hook does NOT use the centralized useApi hook because:
 *
 * 1. **Auto-loading on mount**: This hook fetches data immediately on mount using useEffect,
 *    which is a different pattern from the explicit fetch functions in useApi.
 *
 * 2. **Multiple parallel requests**: The hook fetches both token losses and ollama status
 *    in a single effect, which doesn't map cleanly to the useApi pattern.
 *
 * 3. **Read-only hook**: This is a simple read-only hook with no mutations, making it
 *    lightweight enough that the useApi abstraction doesn't add significant value.
 *
 * 4. **Cleanup handling**: The hook uses a mounted flag for cleanup to prevent state updates
 *    after unmount, which would need to be added to useApi for this use case.
 */
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
