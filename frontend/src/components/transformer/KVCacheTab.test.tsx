import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { KVCacheTab } from "./KVCacheTab";
import type { KVCacheResult } from "../../hooks/transformer/useGlassBoxApi";

const mockKvCache: KVCacheResult = {
  config: {
    context_length: 4096,
    n_layers: 32,
    d_model: 4096,
    n_heads: 32,
    d_head: 128,
    gqa_groups: 8,
    mla_latent_dim: 512,
    bytes_per_param: 2,
  },
  architectures: [
    {
      name: "MHA",
      full_name: "Multi-Head Attention",
      description: "Standard attention",
      memory_bytes: 2147483648,
      memory_formatted: "2.00 GB",
      ratio_to_mha: 1.0,
      kv_heads: 32,
    },
    {
      name: "MQA",
      full_name: "Multi-Query Attention",
      description: "Single shared K/V",
      memory_bytes: 67108864,
      memory_formatted: "64.00 MB",
      ratio_to_mha: 0.03125,
      kv_heads: 1,
    },
    {
      name: "GQA",
      full_name: "Grouped-Query Attention",
      description: "Groups of 8",
      memory_bytes: 268435456,
      memory_formatted: "256.00 MB",
      ratio_to_mha: 0.125,
      kv_heads: 4,
    },
    {
      name: "MLA",
      full_name: "Multi-Head Latent Attention",
      description: "Compressed to 512-dim",
      memory_bytes: 134217728,
      memory_formatted: "128.00 MB",
      ratio_to_mha: 0.0625,
      latent_dim: 512,
    },
  ],
  mha_to_mla_savings: 16.0,
  explanation: "KV Cache comparison",
};

describe("KVCacheTab", () => {
  it("renders placeholder when no data", () => {
    const handleFetch = vi.fn();
    render(
      <KVCacheTab loading={false} kvCache={null} onFetchKvCache={handleFetch} />,
    );

    expect(screen.getByRole("button", { name: "Calculate Memory" })).toBeInTheDocument();
    expect(
      screen.getByText(/Compare memory usage across different attention architectures/),
    ).toBeInTheDocument();
  });

  it("triggers fetch on button click", () => {
    const handleFetch = vi.fn();
    render(
      <KVCacheTab loading={false} kvCache={null} onFetchKvCache={handleFetch} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Calculate Memory" }));
    expect(handleFetch).toHaveBeenCalled();
  });

  it("renders KV cache results with bar chart", () => {
    const handleFetch = vi.fn();
    render(
      <KVCacheTab
        loading={false}
        kvCache={mockKvCache}
        onFetchKvCache={handleFetch}
      />,
    );

    // Architecture names appear in multiple places, so use getAllBy
    expect(screen.getAllByText("MHA").length).toBeGreaterThan(0);
    expect(screen.getAllByText("MQA").length).toBeGreaterThan(0);
    expect(screen.getAllByText("GQA").length).toBeGreaterThan(0);
    expect(screen.getAllByText("MLA").length).toBeGreaterThan(0);

    // Memory values
    expect(screen.getAllByText("2.00 GB").length).toBeGreaterThan(0);
    expect(screen.getAllByText("128.00 MB").length).toBeGreaterThan(0);
  });

  it("shows savings highlight", () => {
    const handleFetch = vi.fn();
    render(
      <KVCacheTab
        loading={false}
        kvCache={mockKvCache}
        onFetchKvCache={handleFetch}
      />,
    );

    expect(screen.getByText("16×")).toBeInTheDocument();
    expect(screen.getByText(/memory reduction from MHA to MLA/)).toBeInTheDocument();
  });

  it("shows educational content", () => {
    const handleFetch = vi.fn();
    render(
      <KVCacheTab
        loading={false}
        kvCache={mockKvCache}
        onFetchKvCache={handleFetch}
      />,
    );

    expect(screen.getByText("Why Does This Matter?")).toBeInTheDocument();
    expect(screen.getByText(/MHA \(Standard\)/)).toBeInTheDocument();
    expect(screen.getByText(/MLA \(DeepSeek R1\)/)).toBeInTheDocument();
  });

  it("has preset selector", () => {
    const handleFetch = vi.fn();
    render(
      <KVCacheTab loading={false} kvCache={null} onFetchKvCache={handleFetch} />,
    );

    expect(screen.getByText("Model Preset")).toBeInTheDocument();
    // Check options are available
    expect(screen.getByRole("combobox")).toBeInTheDocument();
    expect(screen.getByText("Llama 3 8B")).toBeInTheDocument();
  });

  it("disables button when loading", () => {
    const handleFetch = vi.fn();
    render(
      <KVCacheTab loading={true} kvCache={null} onFetchKvCache={handleFetch} />,
    );

    const button = screen.getByRole("button", { name: "Calculating..." });
    expect(button).toBeDisabled();
  });
});
