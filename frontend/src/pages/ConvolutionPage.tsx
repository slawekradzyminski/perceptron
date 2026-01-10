import { useEffect, useState, useCallback } from "react";
import { useConvApi } from "../hooks/conv/useConvApi";
import { ConvControls } from "../components/conv/ConvControls";
import { ConvImageGrid } from "../components/conv/ConvImageGrid";
import { ConvActivationGrid } from "../components/conv/ConvActivationGrid";
import { ConvStepMath } from "../components/conv/ConvStepMath";
import { ConvEducation } from "../components/conv/ConvEducation";
import { ConvDigitRecognition } from "../components/conv/ConvDigitRecognition";
import "../styles/convolution.css";

export function ConvolutionPage({ apiBase }: { apiBase: string }) {
  const {
    state,
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
  } = useConvApi(apiBase);

  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [editingKernel, setEditingKernel] = useState<number[][] | null>(null);

  // Initialize on mount
  useEffect(() => {
    void fetchState();
    void fetchPresets();
  }, [fetchState, fetchPresets]);

  // Fetch step details when hovering
  useEffect(() => {
    if (hoveredCell) {
      void getStep(hoveredCell.row, hoveredCell.col);
    } else {
      clearStep();
    }
  }, [hoveredCell, getStep, clearStep]);

  const handleImageChange = useCallback(
    (name: string) => {
      void setImage(name);
    },
    [setImage],
  );

  const handleKernelChange = useCallback(
    (name: string) => {
      setEditingKernel(null);
      void setKernel(name);
    },
    [setKernel],
  );

  const handlePaddingChange = useCallback(
    (padding: number) => {
      void setParams(padding, undefined);
    },
    [setParams],
  );

  const handleStrideChange = useCallback(
    (stride: number) => {
      void setParams(undefined, stride);
    },
    [setParams],
  );

  const handleImageSizeChange = useCallback(
    (size: number) => {
      void setImageSize(size);
    },
    [setImageSize],
  );

  const handleFileUpload = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        void uploadImage(file);
      }
    },
    [uploadImage],
  );

  const handleKernelWeightChange = (row: number, col: number, value: string) => {
    const numValue = parseFloat(value) || 0;
    const newKernel = editingKernel
      ? editingKernel.map((r) => [...r])
      : state?.kernel.map((r) => [...r]) || [
          [0, 0, 0],
          [0, 1, 0],
          [0, 0, 0],
        ];
    newKernel[row][col] = numValue;
    setEditingKernel(newKernel);
  };

  const applyCustomKernel = () => {
    if (editingKernel) {
      void setKernel(undefined, editingKernel);
      setEditingKernel(null);
    }
  };

  const currentKernel = editingKernel || state?.kernel || [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
  ];

  // Calculate receptive field bounds in the input image
  const getReceptiveFieldBounds = () => {
    if (!step || !state) return null;
    const { input_row, input_col, kernel_size, padding } = step;
    return {
      startRow: input_row - padding,
      startCol: input_col - padding,
      endRow: input_row - padding + kernel_size,
      endCol: input_col - padding + kernel_size,
    };
  };

  const receptiveField = hoveredCell ? getReceptiveFieldBounds() : null;

  return (
    <section className="panel conv-panel">
      <header className="conv-header">
        <h2>Convolution Lab</h2>
        <p className="subtitle">
          Understanding how CNNs see — weight sharing and local connectivity
        </p>
      </header>

      <div className="conv-grid">
        <ConvControls
          state={state}
          loading={loading}
          currentKernel={currentKernel}
          editingKernel={editingKernel}
          onImageChange={handleImageChange}
          onImageSizeChange={handleImageSizeChange}
          onFileUpload={handleFileUpload}
          onKernelChange={handleKernelChange}
          onKernelWeightChange={handleKernelWeightChange}
          onApplyCustomKernel={applyCustomKernel}
          onPaddingChange={handlePaddingChange}
          onStrideChange={handleStrideChange}
        />

        <div className="conv-main">
          <div className="conv-visualization">
            {state && (
              <ConvImageGrid
                image={state.image}
                imageSize={state.image_size}
                imageName={state.image_name}
                inputShape={state.input_shape}
                receptiveField={receptiveField}
              />
            )}

            <div className="conv-arrow">
              <span className="arrow-symbol">→</span>
              <span className="arrow-label">Convolve</span>
            </div>

            {state && (
              <ConvActivationGrid
                activationMap={state.activation_map}
                activationMapNormalized={state.activation_map_normalized}
                outputShape={state.output_shape}
                hoveredCell={hoveredCell}
                onCellHover={setHoveredCell}
              />
            )}
          </div>

          <ConvStepMath step={step} />

          <ConvDigitRecognition apiBase={apiBase} />

          <ConvEducation />
        </div>
      </div>

      {error && <p className="conv-error">{error}</p>}
    </section>
  );
}
