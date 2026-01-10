import { useRef } from "react";
import type {
  AlexNetState,
  FilterGrid,
} from "../../hooks/alexnet/useAlexNetApi";

type AlexNetControlsProps = {
  layers: number[];
  samples: string[];
  selectedLayer: number;
  selectedSample: string;
  layerInfo?: FilterGrid["layer_info"];
  inputImage?: string;
  customImageUrl: string | null;
  state: AlexNetState | null;
  loading: boolean;
  onLayerChange: (layer: number) => void;
  onSampleChange: (sample: string) => void;
  onFileUpload: (file: File) => void;
};

export function AlexNetControls({
  layers,
  samples,
  selectedLayer,
  selectedSample,
  layerInfo,
  inputImage,
  customImageUrl,
  state,
  loading,
  onLayerChange,
  onSampleChange,
  onFileUpload,
}: AlexNetControlsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  };

  return (
    <div className="alexnet-controls">
      <div className="control-group">
        <label>Layer</label>
        <div className="layer-buttons">
          {layers.map((layer) => (
            <button
              key={layer}
              className={selectedLayer === layer ? "active" : ""}
              onClick={() => onLayerChange(layer)}
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
            <strong>
              {layerInfo.kernel_size}×{layerInfo.kernel_size}
            </strong>
          </p>
          <p>Input channels: {layerInfo.in_channels}</p>
        </div>
      )}

      <div className="control-group">
        <label>Test Image</label>
        <div className="sample-buttons">
          {samples.map((sample) => (
            <button
              key={sample}
              className={selectedSample === sample ? "active" : ""}
              onClick={() => onSampleChange(sample)}
              disabled={loading}
            >
              {sample}
            </button>
          ))}
        </div>
        <button className="upload-btn" onClick={handleUploadClick} disabled={loading}>
          Upload Image
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: "none" }}
          data-testid="alexnet-upload-input"
        />
        <p className="upload-hint">Any JPEG/PNG image works. Resized to 224×224.</p>
      </div>

      {inputImage && (
        <div className="input-preview">
          <label>Current Input</label>
          <div className="preview-image">
            <img
              src={customImageUrl || `data:image/png;base64,${inputImage}`}
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
            <strong>{(state.param_count / 1e6).toFixed(1)}M</strong> parameters
          </p>
          <p>5 convolutional layers</p>
        </div>
      )}
    </div>
  );
}
