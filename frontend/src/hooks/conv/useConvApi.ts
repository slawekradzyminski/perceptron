import { useCallback, useState } from "react";

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

export function useConvApi(apiBase: string) {
  const [state, setState] = useState<ConvState | null>(null);
  const [presets, setPresets] = useState<ConvPresets | null>(null);
  const [step, setStep] = useState<ConvStepResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchState = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/conv/state`);
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as ConvState;
      setState(data);
      setError(null);
    } catch {
      setError("API unreachable. Check backend.");
    }
  }, [apiBase]);

  const fetchPresets = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/conv/presets`);
      if (!res.ok) return;
      const data = (await res.json()) as ConvPresets;
      setPresets(data);
    } catch {
      // Silently ignore
    }
  }, [apiBase]);

  const setImage = useCallback(
    async (name: string) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/conv/image/${name}`, {
          method: "POST",
        });
        if (!res.ok) {
          const text = await res.text();
          setError(`API error: ${res.status} - ${text}`);
          return;
        }
        const data = (await res.json()) as ConvState;
        setState(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const setImageSize = useCallback(
    async (size: number) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/conv/image/size`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ size }),
        });
        if (!res.ok) {
          const text = await res.text();
          setError(`API error: ${res.status} - ${text}`);
          return;
        }
        const data = (await res.json()) as ConvState;
        setState(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const uploadImage = useCallback(
    async (file: File) => {
      setLoading(true);
      setError(null);
      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`${apiBase}/conv/image/upload`, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) {
          const text = await res.text();
          setError(`API error: ${res.status} - ${text}`);
          return;
        }
        const data = (await res.json()) as ConvState;
        setState(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const setKernel = useCallback(
    async (name?: string, weights?: number[][]) => {
      setLoading(true);
      setError(null);
      try {
        const body: { name?: string; weights?: number[][] } = {};
        if (name) body.name = name;
        if (weights) body.weights = weights;

        const res = await fetch(`${apiBase}/conv/kernel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const text = await res.text();
          setError(`API error: ${res.status} - ${text}`);
          return;
        }
        const data = (await res.json()) as ConvState;
        setState(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const setParams = useCallback(
    async (padding?: number, stride?: number) => {
      setLoading(true);
      setError(null);
      try {
        const body: { padding?: number; stride?: number } = {};
        if (padding !== undefined) body.padding = padding;
        if (stride !== undefined) body.stride = stride;

        const res = await fetch(`${apiBase}/conv/params`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const text = await res.text();
          setError(`API error: ${res.status} - ${text}`);
          return;
        }
        const data = (await res.json()) as ConvState;
        setState(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const getStep = useCallback(
    async (row: number, col: number) => {
      try {
        const res = await fetch(`${apiBase}/conv/step?row=${row}&col=${col}`);
        if (!res.ok) return;
        const data = (await res.json()) as ConvStepResult;
        setStep(data);
      } catch {
        // Silently ignore
      }
    },
    [apiBase],
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
  };
}
