import { useState, useCallback, useRef, useEffect } from "react";

type ActivationData = {
  name: string;
  shape: number[];
  image: string;
};

type RecognitionResponse = {
  activations: ActivationData[];
  prediction: number;
  confidence: number;
  probabilities: { digit: number; probability: number }[];
};

type ConvDigitRecognitionProps = {
  apiBase: string;
};

export function ConvDigitRecognition({ apiBase }: ConvDigitRecognitionProps) {
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testImage, setTestImage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load pretrained model on mount (ensures it's ready for first request)
  useEffect(() => {
    const loadModel = async () => {
      try {
        await fetch(`${apiBase}/conv/digit/load`, { method: "POST" });
      } catch {
        // Model will be loaded on first recognition request
      }
    };
    void loadModel();
  }, [apiBase]);

  const handleImageUpload = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      // Preview
      const reader = new FileReader();
      reader.onload = (e) => setTestImage(e.target?.result as string);
      reader.readAsDataURL(file);

      // Get recognition results
      setLoading(true);
      setError(null);
      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`${apiBase}/conv/digit/recognize`, {
          method: "POST",
          body: formData,
        });

        if (res.ok) {
          const data = (await res.json()) as RecognitionResponse;
          setResult(data);
        } else {
          setError("Recognition failed");
        }
      } catch {
        setError("Failed to recognize digit");
      } finally {
        setLoading(false);
      }
    },
    [apiBase],
  );

  return (
    <div className="digit-recognition-panel">
      <h3>🔢 Digit Recognition</h3>
      <p className="panel-description">
        Upload a handwritten digit (0-9) to see what the CNN predicts and how
        it processes the image through each layer.
      </p>

      <div className="upload-section">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={loading}
          className="upload-btn"
        >
          {loading ? "Processing..." : "Upload Digit Image"}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          style={{ display: "none" }}
        />
        <p className="upload-hint">
          Best results with clear, centered digits on white background (28×28 px ideal)
        </p>
      </div>

      {testImage && (
        <div className="test-preview">
          <h4>Input Image</h4>
          <img src={testImage} alt="Uploaded digit" />
        </div>
      )}

      {result && (
        <div className="recognition-result">
          {/* Main prediction */}
          <div className="prediction-box">
            <span className="pred-label">Predicted Digit:</span>
            <span className="pred-value">{result.prediction}</span>
            <span className="pred-conf">
              ({(result.confidence * 100).toFixed(1)}% confident)
            </span>
          </div>

          {/* Probability distribution */}
          <div className="probability-section">
            <h4>Probability Distribution</h4>
            <div className="probability-bars">
              {result.probabilities.map(({ digit, probability }) => (
                <div key={digit} className="prob-row">
                  <span className="prob-digit">{digit}</span>
                  <div className="prob-bar-container">
                    <div
                      className="prob-bar"
                      style={{
                        width: `${Math.max(probability * 100, 1)}%`,
                        backgroundColor:
                          digit === result.prediction
                            ? "#16a34a"
                            : probability > 0.1
                              ? "#64748b"
                              : "#e2e8f0",
                      }}
                    />
                  </div>
                  <span className="prob-pct">{(probability * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Layer activations */}
          <div className="layer-activations">
            <h4>Layer Activations</h4>
            <p className="section-hint">
              See how the CNN transforms the input through each layer
            </p>
            <div className="activations-grid">
              {result.activations.map((act, idx) => (
                <div key={idx} className="activation-item">
                  <img
                    src={`data:image/png;base64,${act.image}`}
                    alt={act.name}
                  />
                  <span className="act-name">{act.name}</span>
                  <span className="act-shape">
                    {act.shape.join("×")}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {error && <p className="recognition-error">{error}</p>}
    </div>
  );
}
