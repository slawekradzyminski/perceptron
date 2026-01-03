import { useCallback, useState } from "react";
import type { CustomConfig, ErrorSurfaceResponse, MlpInternalsResponse } from "../types";

export type ErrorSurfaceConfig = {
  steps: number;
  wMin: number;
  wMax: number;
  bias: number;
};

export type MlpInternalsConfig = {
  hiddenDim: number;
  sampleIndex: number;
  lr: number;
  seed: number;
};

function buildDatasetPayload(datasetName: string, customConfig: CustomConfig) {
  if (datasetName !== "custom") {
    return { dataset: datasetName };
  }
  return {
    dataset: "custom",
    grid_rows: customConfig.rows,
    grid_cols: customConfig.cols,
    samples: customConfig.samples,
  };
}

export function useDiagnosticsApi(
  apiBase: string,
  datasetName: string,
  customConfig: CustomConfig,
) {
  const [errorSurface, setErrorSurface] = useState<ErrorSurfaceResponse | null>(null);
  const [errorSurfaceError, setErrorSurfaceError] = useState<string | null>(null);
  const [errorSurfaceLoading, setErrorSurfaceLoading] = useState(false);

  const [mlpInternals, setMlpInternals] = useState<MlpInternalsResponse | null>(null);
  const [mlpInternalsError, setMlpInternalsError] = useState<string | null>(null);
  const [mlpInternalsLoading, setMlpInternalsLoading] = useState(false);

  const fetchErrorSurface = useCallback(
    async (config: ErrorSurfaceConfig) => {
      setErrorSurfaceLoading(true);
      setErrorSurfaceError(null);
      try {
        const payload = {
          ...buildDatasetPayload(datasetName, customConfig),
          steps: config.steps,
          w_min: config.wMin,
          w_max: config.wMax,
          b: config.bias,
        };
        const res = await fetch(`${apiBase}/error-surface`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          setErrorSurfaceError(`API error: ${res.status}`);
          setErrorSurface(null);
          return;
        }
        const data = (await res.json()) as ErrorSurfaceResponse;
        setErrorSurface(data);
      } catch (err) {
        setErrorSurfaceError("API unreachable. Check backend.");
        setErrorSurface(null);
      } finally {
        setErrorSurfaceLoading(false);
      }
    },
    [apiBase, datasetName, customConfig],
  );

  const fetchMlpInternals = useCallback(
    async (config: MlpInternalsConfig) => {
      setMlpInternalsLoading(true);
      setMlpInternalsError(null);
      try {
        const payload = {
          ...buildDatasetPayload(datasetName, customConfig),
          hidden_dim: config.hiddenDim,
          sample_index: config.sampleIndex,
          lr: config.lr,
          seed: config.seed,
        };
        const res = await fetch(`${apiBase}/mlp-internals`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          setMlpInternalsError(`API error: ${res.status}`);
          setMlpInternals(null);
          return;
        }
        const data = (await res.json()) as MlpInternalsResponse;
        setMlpInternals(data);
      } catch (err) {
        setMlpInternalsError("API unreachable. Check backend.");
        setMlpInternals(null);
      } finally {
        setMlpInternalsLoading(false);
      }
    },
    [apiBase, datasetName, customConfig],
  );

  return {
    errorSurface,
    errorSurfaceError,
    errorSurfaceLoading,
    fetchErrorSurface,
    mlpInternals,
    mlpInternalsError,
    mlpInternalsLoading,
    fetchMlpInternals,
  };
}
