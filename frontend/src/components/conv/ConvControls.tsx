import { useRef } from "react";
import type { ConvState } from "../../hooks/conv/useConvApi";

const PRESET_KERNELS = [
  "identity",
  "edge_horizontal",
  "edge_vertical",
  "sobel_x",
  "sobel_y",
  "blur",
  "sharpen",
  "emboss",
  "laplacian",
];

const SAMPLE_IMAGES = [
  "gradient",
  "checkerboard",
  "cross",
  "edges",
  "diagonal",
  "corner",
  "digit_0",
  "digit_1",
];

type ConvControlsProps = {
  state: ConvState | null;
  loading: boolean;
  currentKernel: number[][];
  editingKernel: number[][] | null;
  onImageChange: (name: string) => void;
  onImageSizeChange: (size: number) => void;
  onFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onKernelChange: (name: string) => void;
  onKernelWeightChange: (row: number, col: number, value: string) => void;
  onApplyCustomKernel: () => void;
  onPaddingChange: (padding: number) => void;
  onStrideChange: (stride: number) => void;
};

export function ConvControls({
  state,
  loading,
  currentKernel,
  editingKernel,
  onImageChange,
  onImageSizeChange,
  onFileUpload,
  onKernelChange,
  onKernelWeightChange,
  onApplyCustomKernel,
  onPaddingChange,
  onStrideChange,
}: ConvControlsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="conv-controls">
      {/* Image selection */}
      <div className="control-group">
        <label>Sample Image</label>
        <div className="sample-buttons">
          {SAMPLE_IMAGES.map((img) => (
            <button
              key={img}
              className={state?.image_name === img ? "active" : ""}
              onClick={() => onImageChange(img)}
              disabled={loading}
            >
              {img.replace("_", " ")}
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
          onChange={onFileUpload}
          style={{ display: "none" }}
        />
      </div>

      {/* Image size */}
      <div className="control-group">
        <label>Image Size: {state?.image_size}×{state?.image_size}</label>
        <input
          type="range"
          min="4"
          max="16"
          value={state?.image_size || 8}
          onChange={(e) => onImageSizeChange(parseInt(e.target.value))}
          disabled={loading}
        />
      </div>

      {/* Kernel presets */}
      <div className="control-group">
        <label>Kernel Preset</label>
        <div className="kernel-buttons">
          {PRESET_KERNELS.map((k) => (
            <button
              key={k}
              className={state?.kernel_name === k ? "active" : ""}
              onClick={() => onKernelChange(k)}
              disabled={loading}
            >
              {k.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      {/* Kernel editor */}
      <div className="control-group">
        <label>Kernel Editor (3×3)</label>
        <div className="kernel-editor">
          {[0, 1, 2].map((row) => (
            <div key={row} className="kernel-row">
              {[0, 1, 2].map((col) => (
                <input
                  key={col}
                  type="number"
                  step="0.1"
                  value={currentKernel[row]?.[col]?.toFixed(2) || "0"}
                  onChange={(e) => onKernelWeightChange(row, col, e.target.value)}
                  className="kernel-cell"
                />
              ))}
            </div>
          ))}
        </div>
        {editingKernel && (
          <button className="apply-btn" onClick={onApplyCustomKernel}>
            Apply Custom Kernel
          </button>
        )}
      </div>

      {/* Padding & Stride */}
      <div className="control-group">
        <label>Padding: {state?.padding}</label>
        <div className="param-buttons">
          {[0, 1, 2].map((p) => (
            <button
              key={p}
              className={state?.padding === p ? "active" : ""}
              onClick={() => onPaddingChange(p)}
              disabled={loading}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <div className="control-group">
        <label>Stride: {state?.stride}</label>
        <div className="param-buttons">
          {[1, 2, 3].map((s) => (
            <button
              key={s}
              className={state?.stride === s ? "active" : ""}
              onClick={() => onStrideChange(s)}
              disabled={loading}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Current dimensions */}
      {state && (
        <div className="dim-info">
          <h4>Current Dimensions</h4>
          <p>
            Input: <strong>{state.input_shape[0]}×{state.input_shape[1]}</strong>
          </p>
          <p>
            Kernel: <strong>{state.kernel_shape[0]}×{state.kernel_shape[1]}</strong>
          </p>
          <p>
            Output: <strong>{state.output_shape[0]}×{state.output_shape[1]}</strong>
          </p>
        </div>
      )}
    </div>
  );
}
