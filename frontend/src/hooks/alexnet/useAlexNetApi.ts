import { useCallback, useState } from "react";
import { useApi } from "../common/useApi";

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

/**
 * Hook for AlexNet API operations.
 * Uses the centralized useApi hook for consistent error/loading handling.
 */
export function useAlexNetApi(apiBase: string) {
  const { error, setError, loading, get, post, request } = useApi(apiBase);
  const [state, setState] = useState<AlexNetState | null>(null);
  const [filters, setFilters] = useState<FilterGrid | null>(null);
  const [activations, setActivations] = useState<ActivationGrid | null>(null);

  const fetchState = useCallback(async () => {
    const data = await get<AlexNetState>("/alexnet/state");
    if (data) {
      setState(data);
    }
  }, [get]);

  const setLayer = useCallback(
    async (layer: number) => {
      const data = await post<AlexNetState>(`/alexnet/layer/${layer}`);
      if (data) {
        setState(data);
      }
    },
    [post],
  );

  const fetchFilters = useCallback(
    async (layer: number = 1, maxFilters: number = 64) => {
      const data = await get<FilterGrid>(
        `/alexnet/filters?layer=${layer}&max_filters=${maxFilters}`,
      );
      if (data) {
        setFilters(data);
      }
    },
    [get],
  );

  const fetchActivations = useCallback(
    async (
      sample: string = "gradient",
      layer: number = 1,
      maxActivations: number = 64,
    ) => {
      const data = await get<ActivationGrid>(
        `/alexnet/activations?sample=${sample}&layer=${layer}&max_activations=${maxActivations}`,
      );
      if (data) {
        setActivations(data);
      }
    },
    [get],
  );

  const uploadAndGetActivations = useCallback(
    async (file: File, layer: number = 1, maxActivations: number = 64) => {
      // File uploads need special handling - use rawBody to send FormData directly
      const formData = new FormData();
      formData.append("file", file);

      const data = await request<ActivationGrid>(
        `/alexnet/activations?layer=${layer}&max_activations=${maxActivations}`,
        {
          method: "POST",
          body: formData,
          rawBody: true,
        },
      );
      if (data) {
        setActivations(data);
      }
    },
    [request],
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
    setError,
  };
}
