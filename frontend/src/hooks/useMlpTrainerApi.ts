import { useCallback, useState } from "react";
import type { CustomConfig, MlpTrainerResponse, MlpTrainerSnapshot, MlpInternalsResponse } from "../types";
import { buildCustomPayload } from "../utils/custom";

type ResetOptions = {
  datasetName?: string;
  customConfig?: CustomConfig;
  customApplied?: boolean;
  hiddenDim?: number;
  lr?: number;
  seed?: number;
};

export function useMlpTrainerApi(apiBase: string) {
  const [snapshot, setSnapshot] = useState<MlpTrainerSnapshot | null>(null);
  const [lastStep, setLastStep] = useState<MlpInternalsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const resetWithOptions = useCallback(async (options: ResetOptions = {}) => {
    setLoading(true);
    setError(null);
    try {
      const body: Record<string, unknown> = {};
      if (options.datasetName) {
        if (options.datasetName === "custom") {
          if (!options.customApplied || !options.customConfig) {
            setError("Apply the custom dataset before loading MLP.");
            return;
          }
          Object.assign(body, { dataset: "custom" }, buildCustomPayload(options.customConfig));
        } else {
          body.dataset = options.datasetName;
        }
      }
      if (typeof options.hiddenDim === "number") body.hidden_dim = options.hiddenDim;
      if (typeof options.lr === "number") body.lr = options.lr;
      if (typeof options.seed === "number") body.seed = options.seed;
      const res = await fetch(`${apiBase}/mlp/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as MlpTrainerSnapshot;
      setSnapshot(data);
      setLastStep(null);
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
      const res = await fetch(`${apiBase}/mlp/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as MlpTrainerResponse;
      setSnapshot(data);
      setLastStep(data.step ?? null);
    } catch {
      setError("API unreachable. Check backend.");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  return { snapshot, lastStep, error, loading, resetWithOptions, step };
}
