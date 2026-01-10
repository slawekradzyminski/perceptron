import { useEffect, useState, useCallback } from "react";
import { useConvApi } from "../hooks/conv/useConvApi";
import { ConvControls } from "../components/conv/ConvControls";
import { ConvMainPanel } from "../components/conv/ConvMainPanel";
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

  useEffect(() => {
    void fetchState();
    void fetchPresets();
  }, [fetchState, fetchPresets]);

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

        <ConvMainPanel
          state={state}
          step={step}
          apiBase={apiBase}
          hoveredCell={hoveredCell}
          receptiveField={receptiveField}
          onCellHover={setHoveredCell}
        />
      </div>

      {error && <p className="conv-error">{error}</p>}
    </section>
  );
}
