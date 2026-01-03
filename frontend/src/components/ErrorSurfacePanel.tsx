import { useEffect, useMemo, useRef, useState } from "react";
import type { ErrorSurfaceResponse } from "../types";
import { drawHeatmap, setHeatmapCanvasSize } from "../utils/visuals";

export type ErrorSurfaceConfig = {
  steps: number;
  wMin: number;
  wMax: number;
  bias: number;
};

type ErrorSurfacePanelProps = {
  data: ErrorSurfaceResponse | null;
  loading: boolean;
  error: string | null;
  config: ErrorSurfaceConfig;
  canShow: boolean;
  canFetch: boolean;
  onConfigChange: (config: ErrorSurfaceConfig) => void;
};

export function ErrorSurfacePanel({
  data,
  loading,
  error,
  config,
  canShow,
  canFetch,
  onConfigChange,
}: ErrorSurfacePanelProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{ visible: boolean; text: string; x: number; y: number }>({
    visible: false,
    text: "",
    x: 0,
    y: 0,
  });

  const { minValue, maxValue } = useMemo(() => {
    if (!data) return { minValue: 0, maxValue: 1 };
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    data.grid.forEach((row) => {
      row.forEach((val) => {
        min = Math.min(min, val);
        max = Math.max(max, val);
      });
    });
    return { minValue: min, maxValue: max };
  }, [data]);

  useEffect(() => {
    if (!canvasRef.current || !data) return;
    setHeatmapCanvasSize(canvasRef.current, data.steps, data.steps);
    drawHeatmap(canvasRef.current, data.grid, minValue, maxValue);
  }, [data, minValue, maxValue]);

  const updateConfig = (updates: Partial<ErrorSurfaceConfig>) => {
    onConfigChange({ ...config, ...updates });
  };

  const handleMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!data || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const col = Math.max(
      0,
      Math.min(
        data.steps - 1,
        Math.floor(((event.clientX - rect.left) / rect.width) * data.steps),
      ),
    );
    const row = Math.max(
      0,
      Math.min(
        data.steps - 1,
        Math.floor(((event.clientY - rect.top) / rect.height) * data.steps),
      ),
    );
    const wMin = data.w_range[0];
    const wMax = data.w_range[1];
    const w1 = wMin + (wMax - wMin) * (col / (data.steps - 1));
    const w2 = wMin + (wMax - wMin) * (row / (data.steps - 1));
    const loss = data.grid[row][col];
    setHover({
      visible: true,
      text: `w1=${w1.toFixed(2)}  w2=${w2.toFixed(2)}  loss=${loss.toFixed(2)}`,
      x: event.clientX - rect.left + 12,
      y: event.clientY - rect.top + 12,
    });
  };

  return (
    <section className="panel diagnostics-panel error-surface-panel">
      <h2>Error Surface</h2>
      <div className="panel-subtle">Model: Linear • Loss: MSE</div>
      {!canShow && <p className="diag-note">Error surface is only available for 2D inputs.</p>}
      {!canFetch && canShow && <p className="diag-note">Apply the custom dataset to load diagnostics.</p>}
      {error && <p className="diag-error">{error}</p>}
      {loading && canShow && <p className="diag-note">Loading surface...</p>}

      {!canShow && (
        <div className="diag-placeholder">
          <p>Select a 2D dataset (OR/XOR) or a custom 1×2 grid to view the surface.</p>
        </div>
      )}
      {canShow && (
        <>
      <div className="diag-controls">
        <label>
          Steps
          <input
            type="number"
            min={5}
            max={60}
            value={config.steps}
            onChange={(event) => updateConfig({ steps: Number(event.target.value) })}
          />
        </label>
        <div className="diag-row">
          <label>
            W min
            <input
              type="number"
              step="0.5"
              value={config.wMin}
              onChange={(event) => updateConfig({ wMin: Number(event.target.value) })}
            />
          </label>
          <label>
            W max
            <input
              type="number"
              step="0.5"
              value={config.wMax}
              onChange={(event) => updateConfig({ wMax: Number(event.target.value) })}
            />
          </label>
        </div>
        <label>
          Bias
          <input
            type="number"
            step="0.5"
            value={config.bias}
            onChange={(event) => updateConfig({ bias: Number(event.target.value) })}
          />
        </label>
      </div>

      {data && (
        <div className="heatmap-wrap">
          <canvas
            ref={canvasRef}
            aria-label="Error surface heatmap"
            onMouseMove={handleMove}
            onMouseLeave={() => setHover((prev) => ({ ...prev, visible: false }))}
          />
          {hover.visible && (
            <div className="heatmap-tooltip" data-testid="heatmap-tooltip" style={{ left: hover.x, top: hover.y }}>
              {hover.text}
            </div>
          )}
          <div className="heatmap-legend">
            <span>{minValue.toFixed(2)}</span>
            <div className="heatmap-gradient" />
            <span>{maxValue.toFixed(2)}</span>
          </div>
          <p className="diag-note">
            Each pixel is the mean squared error across the dataset for a specific (w₁, w₂) pair.
            Darker regions indicate higher error.
          </p>
        </div>
      )}
      {!data && !loading && !error && canFetch && canShow && (
        <p className="diag-note">Tune the controls to load a surface.</p>
      )}
        </>
      )}
    </section>
  );
}
