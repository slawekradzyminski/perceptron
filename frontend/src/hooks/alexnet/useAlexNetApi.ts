import { useCallback, useState } from "react";

export type LayerInfo = {
  layer: number;
  name: string;
  filters: number;
  kernel_size: number;
  in_channels: number;
};

export type AlexNetState = {
  current_layer: number;
  sample_images: string[];
  layers: LayerInfo[];
  param_count: number;
};

export type FilterItem = {
  index: number;
  kernel_size: number;
  image: string;
};

export type FilterGrid = {
  rows: number;
  cols: number;
  count: number;
  layer: number;
  layer_info: {
    name: string;
    filters: number;
    kernel_size: number;
    in_channels: number;
  };
  filters: FilterItem[];
};

export type ActivationItem = {
  index: number;
  width: number;
  height: number;
  max_activation: number;
  image: string;
};

export type ActivationGrid = {
  rows: number;
  cols: number;
  count: number;
  layer: number;
  layer_info: {
    name: string;
    filters: number;
    kernel_size: number;
    in_channels: number;
  };
  sample_name: string | null;
  input_image: string; // Base64-encoded input image
  activations: ActivationItem[];
};

export function useAlexNetApi(apiBase: string) {
  const [state, setState] = useState<AlexNetState | null>(null);
  const [filters, setFilters] = useState<FilterGrid | null>(null);
  const [activations, setActivations] = useState<ActivationGrid | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchState = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/alexnet/state`);
      if (!res.ok) {
        setError(`API error: ${res.status}`);
        return;
      }
      const data = (await res.json()) as AlexNetState;
      setState(data);
      setError(null);
    } catch {
      setError("API unreachable. Check backend.");
    }
  }, [apiBase]);

  const setLayer = useCallback(
    async (layer: number) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${apiBase}/alexnet/layer/${layer}`, {
          method: "POST",
        });
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const data = (await res.json()) as AlexNetState;
        setState(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const fetchFilters = useCallback(
    async (layer: number = 1, maxFilters: number = 64) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${apiBase}/alexnet/filters?layer=${layer}&max_filters=${maxFilters}`,
        );
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const data = (await res.json()) as FilterGrid;
        setFilters(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const fetchActivations = useCallback(
    async (
      sample: string = "gradient",
      layer: number = 1,
      maxActivations: number = 64,
    ) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${apiBase}/alexnet/activations?sample=${sample}&layer=${layer}&max_activations=${maxActivations}`,
        );
        if (!res.ok) {
          setError(`API error: ${res.status}`);
          return;
        }
        const data = (await res.json()) as ActivationGrid;
        setActivations(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  const uploadAndGetActivations = useCallback(
    async (file: File, layer: number = 1, maxActivations: number = 64) => {
      setLoading(true);
      setError(null);
      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(
          `${apiBase}/alexnet/activations?layer=${layer}&max_activations=${maxActivations}`,
          {
            method: "POST",
            body: formData,
          },
        );
        if (!res.ok) {
          const text = await res.text();
          setError(`API error: ${res.status} - ${text}`);
          return;
        }
        const data = (await res.json()) as ActivationGrid;
        setActivations(data);
      } catch {
        setError("API unreachable. Check backend.");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  return {
    state,
    filters,
    activations,
    error,
    loading,
    fetchState,
    setLayer,
    fetchFilters,
    fetchActivations,
    uploadAndGetActivations,
  };
}

