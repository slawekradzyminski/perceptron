import { useCallback, useState } from "react";
import type { CustomConfig, MlpTrainerResponse, MlpTrainerSnapshot, MlpInternalsResponse } from "../../types";
import { buildCustomPayload } from "../../utils/custom";
import { useApi } from "../common/useApi";

type ResetOptions = {
  datasetName?: string;
  customConfig?: CustomConfig;
  customApplied?: boolean;
  hiddenDim?: number;
  lr?: number;
  seed?: number;
};

/**
 * Hook for MLP Trainer API operations.
 * Uses the centralized useApi hook for consistent error/loading handling.
 */
export function useMlpTrainerApi(apiBase: string) {
  const { error, setError, loading, post } = useApi(apiBase);
  const [snapshot, setSnapshot] = useState<MlpTrainerSnapshot | null>(null);
  const [lastStep, setLastStep] = useState<MlpInternalsResponse | null>(null);

  const resetWithOptions = useCallback(
    async (options: ResetOptions = {}) => {
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

      const data = await post<MlpTrainerSnapshot>("/mlp/reset", body);
      if (data) {
        setSnapshot(data);
        setLastStep(null);
      }
    },
    [post, setError],
  );

  const step = useCallback(async () => {
    const data = await post<MlpTrainerResponse>("/mlp/step");
    if (data) {
      setSnapshot(data);
      setLastStep(data.step ?? null);
    }
  }, [post]);

  return { snapshot, lastStep, error, loading, resetWithOptions, step, setError };
}
