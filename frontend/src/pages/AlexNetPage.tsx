import { useEffect, useState, useCallback } from "react";
import { useAlexNetApi } from "../hooks/alexnet/useAlexNetApi";
import { AlexNetEducation } from "../components/education/AlexNetEducation";
import { AlexNetControls } from "../components/alexnet/AlexNetControls";
import { AlexNetActivationSection } from "../components/alexnet/AlexNetActivationSection";
import { AlexNetFilterSection } from "../components/alexnet/AlexNetFilterSection";
import "../styles/alexnet.css";
import "../styles/education.css";

const LAYERS = [1, 2, 3, 4, 5];
const SAMPLES = [
  "gradient",
  "checkerboard",
  "noise",
  "face",
  "cat",
  "car",
  "landscape",
];

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

  useEffect(() => {
    void fetchState();
    void fetchFilters(1, 64);
    void fetchActivations("gradient", 1, 64);
  }, [fetchState, fetchFilters, fetchActivations]);

  const handleLayerChange = useCallback(
    (layer: number) => {
      if (layer < 1 || layer > 5) return;
      setSelectedLayer(layer);
      void setLayer(layer);
      void fetchFilters(layer, 64);
      if (selectedSample === "custom" && customImageFile) {
        void uploadAndGetActivations(customImageFile, layer, 64);
      } else {
        void fetchActivations(selectedSample, layer, 64);
      }
    },
    [
      setLayer,
      fetchFilters,
      fetchActivations,
      uploadAndGetActivations,
      selectedSample,
      customImageFile,
    ],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
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
      setCustomImageUrl(null);
      setCustomImageFile(null);
      void fetchActivations(sample, selectedLayer, 64);
    },
    [fetchActivations, selectedLayer],
  );

  const handleFileUpload = useCallback(
    (file: File) => {
      setSelectedSample("custom");
      setCustomImageFile(file);
      setCustomImageUrl(URL.createObjectURL(file));
      void uploadAndGetActivations(file, selectedLayer, 64);
    },
    [uploadAndGetActivations, selectedLayer],
  );

  const layerInfo = filters?.layer_info;

  return (
    <section className="panel alexnet-panel">
      <header className="alexnet-header">
        <h2>AlexNet Explorer</h2>
        <p className="subtitle">
          Visualizing Convolutional Neural Networks — See what the network learns
          to detect
        </p>
      </header>

      <div className="alexnet-grid">
        <AlexNetControls
          layers={LAYERS}
          samples={SAMPLES}
          selectedLayer={selectedLayer}
          selectedSample={selectedSample}
          layerInfo={layerInfo}
          inputImage={activations?.input_image}
          customImageUrl={customImageUrl}
          state={state}
          loading={loading}
          onLayerChange={handleLayerChange}
          onSampleChange={handleSampleChange}
          onFileUpload={handleFileUpload}
        />

        <div className="alexnet-main">
          <AlexNetActivationSection
            activations={activations}
            selectedSample={selectedSample}
            customImageUrl={customImageUrl}
            hoveredFilter={hoveredFilter}
          />

          <AlexNetFilterSection
            filters={filters}
            selectedLayer={selectedLayer}
            hoveredFilter={hoveredFilter}
            onHoverChange={setHoveredFilter}
          />
        </div>
      </div>

      <AlexNetEducation />

      {error && <p className="alexnet-error">{error}</p>}
    </section>
  );
}
