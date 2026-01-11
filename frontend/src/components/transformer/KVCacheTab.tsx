import { useState, useCallback, useEffect } from "react";
import type { KVCacheResult } from "../../hooks/transformer/useGlassBoxApi";

type KVCacheTabProps = {
  loading: boolean;
  kvCache: KVCacheResult | null;
  onFetchKvCache: (config: {
    context_length?: number;
    n_layers?: number;
    d_model?: number;
    n_heads?: number;
    gqa_groups?: number;
    mla_latent_dim?: number;
  }) => void;
};

// Preset configurations for different models
const MODEL_PRESETS = {
  "GPT-2 Small": {
    context_length: 1024,
    n_layers: 12,
    d_model: 768,
    n_heads: 12,
    gqa_groups: 1,
    mla_latent_dim: 192,
  },
  "Llama 3 8B": {
    context_length: 8192,
    n_layers: 32,
    d_model: 4096,
    n_heads: 32,
    gqa_groups: 4,
    mla_latent_dim: 512,
  },
  "DeepSeek R1": {
    context_length: 65536,
    n_layers: 64,
    d_model: 8192,
    n_heads: 64,
    gqa_groups: 8,
    mla_latent_dim: 1024,
  },
  Custom: {
    context_length: 4096,
    n_layers: 32,
    d_model: 4096,
    n_heads: 32,
    gqa_groups: 8,
    mla_latent_dim: 512,
  },
};

export function KVCacheTab({
  loading,
  kvCache,
  onFetchKvCache,
}: KVCacheTabProps) {
  const [selectedPreset, setSelectedPreset] =
    useState<keyof typeof MODEL_PRESETS>("Llama 3 8B");
  const [config, setConfig] = useState(MODEL_PRESETS["Llama 3 8B"]);

  // Update config when preset changes
  useEffect(() => {
    setConfig(MODEL_PRESETS[selectedPreset]);
  }, [selectedPreset]);

  const handleCalculate = useCallback(() => {
    onFetchKvCache(config);
  }, [config, onFetchKvCache]);

  const handleConfigChange = (field: keyof typeof config, value: number) => {
    setSelectedPreset("Custom");
    setConfig((prev) => ({ ...prev, [field]: value }));
  };

  // Calculate max memory for bar chart scaling
  const maxMemory = kvCache
    ? Math.max(...kvCache.architectures.map((a) => a.memory_bytes))
    : 1;

  // Color mapping for architectures
  const archColors: Record<string, string> = {
    MHA: "#7c3aed",
    MQA: "#2563eb",
    GQA: "#059669",
    MLA: "#dc2626",
  };

  return (
    <div className="kv-cache-tab">
      <div className="kv-cache-header">
        <h3>KV Cache Memory Calculator</h3>
        <p>
          Compare memory usage across different attention architectures
          (Chapter 8).
        </p>
      </div>

      <div className="kv-cache-controls">
        <div className="preset-selector">
          <label>Model Preset</label>
          <select
            value={selectedPreset}
            onChange={(e) =>
              setSelectedPreset(e.target.value as keyof typeof MODEL_PRESETS)
            }
          >
            {Object.keys(MODEL_PRESETS).map((preset) => (
              <option key={preset} value={preset}>
                {preset}
              </option>
            ))}
          </select>
        </div>

        <div className="config-grid">
          <div className="config-item">
            <label>Context Length</label>
            <input
              type="number"
              value={config.context_length}
              onChange={(e) =>
                handleConfigChange("context_length", Number(e.target.value))
              }
              min={1}
              max={1000000}
            />
          </div>

          <div className="config-item">
            <label>Layers</label>
            <input
              type="number"
              value={config.n_layers}
              onChange={(e) =>
                handleConfigChange("n_layers", Number(e.target.value))
              }
              min={1}
              max={200}
            />
          </div>

          <div className="config-item">
            <label>Hidden Dim (d_model)</label>
            <input
              type="number"
              value={config.d_model}
              onChange={(e) =>
                handleConfigChange("d_model", Number(e.target.value))
              }
              min={64}
              max={65536}
            />
          </div>

          <div className="config-item">
            <label>Heads</label>
            <input
              type="number"
              value={config.n_heads}
              onChange={(e) =>
                handleConfigChange("n_heads", Number(e.target.value))
              }
              min={1}
              max={256}
            />
          </div>

          <div className="config-item">
            <label>GQA Groups</label>
            <input
              type="number"
              value={config.gqa_groups}
              onChange={(e) =>
                handleConfigChange("gqa_groups", Number(e.target.value))
              }
              min={1}
              max={config.n_heads}
            />
          </div>

          <div className="config-item">
            <label>MLA Latent Dim</label>
            <input
              type="number"
              value={config.mla_latent_dim}
              onChange={(e) =>
                handleConfigChange("mla_latent_dim", Number(e.target.value))
              }
              min={1}
              max={config.d_model}
            />
          </div>
        </div>

        <button onClick={handleCalculate} disabled={loading}>
          {loading ? "Calculating..." : "Calculate Memory"}
        </button>
      </div>

      {kvCache && (
        <div className="kv-cache-results">
          <div className="memory-chart">
            {kvCache.architectures.map((arch) => (
              <div key={arch.name} className="memory-bar-row">
                <div className="arch-info">
                  <span className="arch-name">{arch.name}</span>
                  <span className="arch-full-name">{arch.full_name}</span>
                </div>
                <div className="bar-container">
                  <div
                    className="memory-bar"
                    style={{
                      width: `${(arch.memory_bytes / maxMemory) * 100}%`,
                      backgroundColor: archColors[arch.name] || "#666",
                    }}
                  />
                  <span className="memory-value">{arch.memory_formatted}</span>
                </div>
                <div className="ratio-badge">
                  {arch.ratio_to_mha === 1
                    ? "baseline"
                    : `${(arch.ratio_to_mha * 100).toFixed(1)}%`}
                </div>
              </div>
            ))}
          </div>

          <div className="kv-cache-summary">
            <div className="savings-highlight">
              <span className="savings-number">
                {kvCache.mha_to_mla_savings}×
              </span>
              <span className="savings-label">
                memory reduction from MHA to MLA
              </span>
            </div>
            <p className="explanation">{kvCache.explanation}</p>
          </div>

          <div className="kv-cache-details">
            <h4>Architecture Details</h4>
            <table className="details-table">
              <thead>
                <tr>
                  <th>Architecture</th>
                  <th>Description</th>
                  <th>KV Heads</th>
                  <th>Memory</th>
                </tr>
              </thead>
              <tbody>
                {kvCache.architectures.map((arch) => (
                  <tr key={arch.name}>
                    <td>
                      <strong>{arch.name}</strong>
                    </td>
                    <td>{arch.description}</td>
                    <td>
                      {arch.kv_heads !== undefined
                        ? arch.kv_heads
                        : `latent: ${arch.latent_dim}`}
                    </td>
                    <td>{arch.memory_formatted}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="kv-cache-education">
            <h4>Why Does This Matter?</h4>
            <ul>
              <li>
                <strong>MHA (Standard)</strong>: Each head stores its own K/V —
                expensive for long contexts.
              </li>
              <li>
                <strong>MQA</strong>: All heads share one K/V — extreme
                compression but may lose quality.
              </li>
              <li>
                <strong>GQA (Llama 3)</strong>: Groups of heads share K/V — good
                balance of efficiency and quality.
              </li>
              <li>
                <strong>MLA (DeepSeek R1)</strong>: Compresses K/V into a
                learned latent space — best of both worlds.
              </li>
            </ul>
            <p className="hint">
              DeepSeek's MLA enables 64K+ context windows with manageable memory
              (Chapter 8).
            </p>
          </div>
        </div>
      )}

      {!kvCache && !loading && (
        <div className="kv-cache-placeholder">
          <p>
            Configure model parameters above and click "Calculate Memory" to
            compare attention architectures.
          </p>
          <p className="hint">
            This demonstrates why MLA enables much longer context windows than
            traditional attention (Chapter 8).
          </p>
        </div>
      )}
    </div>
  );
}
