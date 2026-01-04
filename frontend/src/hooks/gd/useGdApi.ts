import { useEffect, useState } from "react";
import type { GdLossCurvesResponse, TokenLossResponse } from "../../types";

export function useGdApi(apiBase: string) {
  const [lossCurves, setLossCurves] = useState<GdLossCurvesResponse | null>(null);
  const [tokenLosses, setTokenLosses] = useState<TokenLossResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const curvesRes = await fetch(`${apiBase}/gd/loss-curves`);
        if (!curvesRes.ok) throw new Error("Failed to load loss curves");
        const curvesData = (await curvesRes.json()) as GdLossCurvesResponse;

        const tokenRes = await fetch(`${apiBase}/gd/token-losses`);
        if (!tokenRes.ok) throw new Error("Failed to load token losses");
        const tokenData = (await tokenRes.json()) as TokenLossResponse;

        if (mounted) {
          setLossCurves(curvesData);
          setTokenLosses(tokenData);
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

  return { lossCurves, tokenLosses, error, loading };
}
