import { useCallback, useState } from "react";
import { useApi } from "../common/useApi";

export type ConvState = {
  image: number[][];
  image_name: string;
  image_size: number;
  kernel: number[][];
  kernel_name: string;
  padding: number;
  stride: number;
  input_shape: [number, number];
  kernel_shape: [number, number];
  output_shape: [number, number];
  activation_map: number[][];
  activation_map_normalized: number[][];
  image_base64: string;
  activation_base64: string;
};

export type ConvPresets = {
  images: string[];
  kernels: string[];
  kernel_weights: Record<string, number[][]>;
};

export type ConvStepResult = {
  output_row: number;
  output_col: number;
  input_row: number;
  input_col: number;
  patch: number[][];
  kernel: number[][];
  products: number[][];
  sum: number;
  kernel_size: number;
  padding: number;
  stride: number;
};

/**
 * Hook for Convolution visualization API operations.
 * Uses the centralized useApi hook for consistent error/loading handling.
 */
export function useConvApi(apiBase: string) {
  const { error, setError, loading, get, post, request } = useApi(apiBase);
  const [state, setState] = useState<ConvState | null>(null);
  const [presets, setPresets] = useState<ConvPresets | null>(null);
  const [step, setStep] = useState<ConvStepResult | null>(null);

  const fetchState = useCallback(async () => {
    const data = await get<ConvState>("/conv/state");
    if (data) {
      setState(data);
    }
  }, [get]);

  const fetchPresets = useCallback(async () => {
    const data = await get<ConvPresets>("/conv/presets");
    if (data) {
      setPresets(data);
    }
  }, [get]);

  const setImage = useCallback(
    async (name: string) => {
      const data = await post<ConvState>(`/conv/image/${name}`);
      if (data) {
        setState(data);
      }
    },
    [post],
  );

  const setImageSize = useCallback(
    async (size: number) => {
      const data = await post<ConvState>("/conv/image/size", { size });
      if (data) {
        setState(data);
      }
    },
    [post],
  );

  const uploadImage = useCallback(
    async (file: File) => {
      // File uploads need rawBody for FormData
      const formData = new FormData();
      formData.append("file", file);

      const data = await request<ConvState>("/conv/image/upload", {
        method: "POST",
        body: formData,
        rawBody: true,
      });
      if (data) {
        setState(data);
      }
    },
    [request],
  );

  const setKernel = useCallback(
    async (name?: string, weights?: number[][]) => {
      const body: { name?: string; weights?: number[][] } = {};
      if (name) body.name = name;
      if (weights) body.weights = weights;

      const data = await post<ConvState>("/conv/kernel", body);
      if (data) {
        setState(data);
      }
    },
    [post],
  );

  const setParams = useCallback(
    async (padding?: number, stride?: number) => {
      const body: { padding?: number; stride?: number } = {};
      if (padding !== undefined) body.padding = padding;
      if (stride !== undefined) body.stride = stride;

      const data = await post<ConvState>("/conv/params", body);
      if (data) {
        setState(data);
      }
    },
    [post],
  );

  const getStep = useCallback(
    async (row: number, col: number) => {
      const data = await get<ConvStepResult>(`/conv/step?row=${row}&col=${col}`);
      if (data) {
        setStep(data);
      }
    },
    [get],
  );

  const clearStep = useCallback(() => {
    setStep(null);
  }, []);

  return {
    state,
    presets,
    step,
    error,
    loading,
    fetchState,
    fetchPresets,
    setImage,
    setImageSize,
    uploadImage,
    setKernel,
    setParams,
    getStep,
    clearStep,
    setError,
  };
}
