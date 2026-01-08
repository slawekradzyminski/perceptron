import { useEffect, useState, useCallback, useRef } from "react";
import { useAlexNetApi } from "../hooks/alexnet/useAlexNetApi";
import "../styles/alexnet.css";

const LAYERS = [1, 2, 3, 4, 5];
const SAMPLES = ["gradient", "checkerboard", "noise", "face", "cat", "car", "landscape"];

export function AlexNetPage({ apiBase }: { apiBase: string }) {
  const {
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
  } = useAlexNetApi(apiBase);

  const [selectedLayer, setSelectedLayer] = useState(1);
  const [selectedSample, setSelectedSample] = useState("gradient");
  const [hoveredFilter, setHoveredFilter] = useState<number | null>(null);
  const [customImageUrl, setCustomImageUrl] = useState<string | null>(null);
  const [customImageFile, setCustomImageFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize on mount
  useEffect(() => {
    void fetchState();
    void fetchFilters(1, 64);
    void fetchActivations("gradient", 1, 64);
  }, [fetchState, fetchFilters, fetchActivations]);

  const handleLayerChange = useCallback(
    (layer: number) => {
      if (layer < 1 || layer > 5) return; // Ignore invalid layers
      setSelectedLayer(layer);
      void setLayer(layer);
      void fetchFilters(layer, 64);
      // If we have a custom image, re-upload it for the new layer
      if (selectedSample === "custom" && customImageFile) {
        void uploadAndGetActivations(customImageFile, layer, 64);
      } else {
        void fetchActivations(selectedSample, layer, 64);
      }
    },
    [setLayer, fetchFilters, fetchActivations, uploadAndGetActivations, selectedSample, customImageFile],
  );

  // Keyboard navigation for layers (arrow keys)
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Don't intercept if user is typing in an input
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setSelectedLayer((current) => {
          const newLayer = current - 1;
          if (newLayer >= 1) {
            handleLayerChange(newLayer);
            return newLayer;
          }
          return current;
        });
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setSelectedLayer((current) => {
          const newLayer = current + 1;
          if (newLayer <= 5) {
            handleLayerChange(newLayer);
            return newLayer;
          }
          return current;
        });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleLayerChange]);

  const handleSampleChange = useCallback(
    (sample: string) => {
      setSelectedSample(sample);
      setCustomImageUrl(null); // Clear custom image when switching to samples
      setCustomImageFile(null);
      void fetchActivations(sample, selectedLayer, 64);
    },
    [fetchActivations, selectedLayer],
  );

  const handleFileUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        setSelectedSample("custom");
        // Store the file for re-use when changing layers
        setCustomImageFile(file);
        // Create local URL for preview of custom uploads
        setCustomImageUrl(URL.createObjectURL(file));
        void uploadAndGetActivations(file, selectedLayer, 64);
      }
    },
    [uploadAndGetActivations, selectedLayer],
  );

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const layerInfo = filters?.layer_info;

  return (
    <section className="panel alexnet-panel">
      <header className="alexnet-header">
        <h2>AlexNet Explorer</h2>
        <p className="subtitle">
          Visualizing Convolutional Neural Networks — See what the network
          learns to detect
        </p>
      </header>

      <div className="alexnet-grid">
        {/* Controls */}
        <div className="alexnet-controls">
          <div className="control-group">
            <label>Layer</label>
            <div className="layer-buttons">
              {LAYERS.map((layer) => (
                <button
                  key={layer}
                  className={selectedLayer === layer ? "active" : ""}
                  onClick={() => handleLayerChange(layer)}
                  disabled={loading}
                >
                  {layer}
                </button>
              ))}
            </div>
          </div>

          {layerInfo && (
            <div className="layer-info">
              <h4>{layerInfo.name}</h4>
              <p>
                <strong>{layerInfo.filters}</strong> filters ×{" "}
                <strong>{layerInfo.kernel_size}×{layerInfo.kernel_size}</strong>
              </p>
              <p>Input channels: {layerInfo.in_channels}</p>
            </div>
          )}

          <div className="control-group">
            <label>Test Image</label>
            <div className="sample-buttons">
              {SAMPLES.map((sample) => (
                <button
                  key={sample}
                  className={selectedSample === sample ? "active" : ""}
                  onClick={() => handleSampleChange(sample)}
                  disabled={loading}
                >
                  {sample}
                </button>
              ))}
            </div>
            <button
              className="upload-btn"
              onClick={handleUploadClick}
              disabled={loading}
            >
              Upload Image
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              style={{ display: "none" }}
            />
            <p className="upload-hint">
              Any JPEG/PNG image works. Resized to 224×224.
            </p>
          </div>

          {/* Input Image Preview */}
          {activations?.input_image && (
            <div className="input-preview">
              <label>Current Input</label>
              <div className="preview-image">
                <img
                  src={
                    customImageUrl ||
                    `data:image/png;base64,${activations.input_image}`
                  }
                  alt={`Input: ${selectedSample}`}
                />
              </div>
              <span className="preview-label">
                {selectedSample === "custom" ? "Uploaded" : selectedSample}
              </span>
            </div>
          )}

          {state && (
            <div className="model-info">
              <h4>AlexNet</h4>
              <p>
                <strong>{(state.param_count / 1e6).toFixed(1)}M</strong>{" "}
                parameters
              </p>
              <p>5 convolutional layers</p>
            </div>
          )}
        </div>

        {/* Main visualization */}
        <div className="alexnet-main">
          {/* Activations grid - shown first (on top) */}
          {activations && (
            <div className="viz-section">
              <h3>
                Activation Maps
                <span className="hint">
                  {" "}
                  — Where each filter fires on the input
                </span>
              </h3>

              {/* Input reference in activation section */}
              <div className="activation-header">
                <div className="input-reference">
                  <span className="ref-label">Analyzing:</span>
                  <img
                    src={
                      customImageUrl ||
                      `data:image/png;base64,${activations.input_image}`
                    }
                    alt="Input being analyzed"
                    className="ref-image"
                  />
                  <span className="ref-name">
                    {selectedSample === "custom" ? "Uploaded image" : selectedSample}
                  </span>
                </div>
              </div>

              <div className="activation-grid">
                {activations.activations.map((act) => (
                  <div
                    key={act.index}
                    className={`activation-cell ${hoveredFilter === act.index ? "highlighted" : ""}`}
                    title={`Filter ${act.index} — max: ${act.max_activation.toFixed(2)}`}
                  >
                    <img
                      src={`data:image/png;base64,${act.image}`}
                      alt={`Activation ${act.index}`}
                    />
                    <span className="activation-index">{act.index}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Filters grid - shown second (at bottom) */}
          <div className="viz-section">
            <h3>
              Pattern Detectors (Layer {selectedLayer} Filters)
              {selectedLayer === 1 && (
                <span className="hint">
                  {" "}
                  — Edge and color blob detectors
                </span>
              )}
              {selectedLayer === 2 && (
                <span className="hint"> — Corner and texture detectors</span>
              )}
              {selectedLayer >= 3 && (
                <span className="hint"> — High-level feature detectors</span>
              )}
            </h3>
            <div className="filter-grid">
              {filters?.filters.map((filter) => (
                <div
                  key={filter.index}
                  className={`filter-cell ${hoveredFilter === filter.index ? "hovered" : ""}`}
                  onMouseEnter={() => setHoveredFilter(filter.index)}
                  onMouseLeave={() => setHoveredFilter(null)}
                  title={`Filter ${filter.index}`}
                >
                  <img
                    src={`data:image/png;base64,${filter.image}`}
                    alt={`Filter ${filter.index}`}
                  />
                  <span className="filter-index">{filter.index}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {error && <p className="alexnet-error">{error}</p>}
    </section>
  );
}

