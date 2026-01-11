/**
 * Generic API Hook Factory
 *
 * This module provides a reusable hook for API calls with consistent
 * loading state, error handling, and fetch patterns.
 */

import { useCallback, useState } from "react";

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

export interface ApiRequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
  /** If true, body is sent as-is (for FormData). Otherwise JSON.stringify is applied. */
  rawBody?: boolean;
}

/**
 * Generic hook for making API requests with consistent loading/error handling.
 *
 * @param apiBase - Base URL for API endpoints
 * @returns Object with request function and state management
 */
export function useApi<T = unknown>(apiBase: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  /**
   * Make an API request with automatic loading/error state management.
   *
   * @param endpoint - API endpoint path (will be appended to apiBase)
   * @param options - Fetch options (method, body, headers, etc.)
   * @returns The response data or null if an error occurred
   */
  const request = useCallback(
    async <R = T>(
      endpoint: string,
      options?: ApiRequestOptions,
    ): Promise<R | null> => {
      setLoading(true);
      setError(null);
      try {
        const { body, rawBody, ...restOptions } = options ?? {};

        // Determine body and headers based on rawBody flag
        let requestBody: BodyInit | undefined;
        let headers: HeadersInit = { "Content-Type": "application/json" };

        if (body !== undefined) {
          if (rawBody) {
            // For FormData or other raw bodies, don't stringify and let browser set Content-Type
            requestBody = body as BodyInit;
            headers = {};
          } else {
            requestBody = JSON.stringify(body);
          }
        }

        const res = await fetch(`${apiBase}${endpoint}`, {
          headers,
          ...restOptions,
          body: requestBody,
        });
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({}));
          const errorMessage =
            errorData.detail || `API error: ${res.status}`;
          setError(errorMessage);
          return null;
        }
        const responseData = (await res.json()) as R;
        return responseData;
      } catch {
        setError("API unreachable. Check backend.");
        return null;
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  /**
   * Make a GET request.
   */
  const get = useCallback(
    async <R = T>(endpoint: string): Promise<R | null> => {
      return request<R>(endpoint, { method: "GET" });
    },
    [request],
  );

  /**
   * Make a POST request.
   */
  const post = useCallback(
    async <R = T>(endpoint: string, body?: unknown): Promise<R | null> => {
      return request<R>(endpoint, { method: "POST", body });
    },
    [request],
  );

  /**
   * Clear error state.
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    data,
    setData,
    error,
    setError,
    loading,
    request,
    get,
    post,
    clearError,
  };
}

/**
 * Simplified hook for a single API resource with auto-fetch capability.
 *
 * @param apiBase - Base URL for API endpoints
 * @param endpoint - Endpoint to fetch
 * @returns Object with state and fetch function
 */
export function useApiResource<T>(apiBase: string, endpoint: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetch = useCallback(async (): Promise<T | null> => {
    setLoading(true);
    setError(null);
    try {
      const res = await globalThis.fetch(`${apiBase}${endpoint}`, {
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        setError(errorData.detail || `API error: ${res.status}`);
        return null;
      }
      const responseData = (await res.json()) as T;
      setData(responseData);
      return responseData;
    } catch {
      setError("API unreachable. Check backend.");
      return null;
    } finally {
      setLoading(false);
    }
  }, [apiBase, endpoint]);

  return { data, setData, error, loading, fetch };
}
