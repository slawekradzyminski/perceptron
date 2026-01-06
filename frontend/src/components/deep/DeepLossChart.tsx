import { useEffect, useRef } from "react";
import type { DeepHistoryEntry } from "../../hooks/deep/useDeepApi";

type DeepLossChartProps = {
  history: DeepHistoryEntry[];
};

export function DeepLossChart({ history }: DeepLossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 30;

    // Clear
    ctx.fillStyle = "#fffdf8";
    ctx.fillRect(0, 0, width, height);

    // Get data range
    const losses = history.map((h) => h.loss);
    const accuracies = history.map((h) => h.accuracy);
    const maxLoss = Math.max(...losses, 0.1);
    const minLoss = Math.min(...losses, 0);

    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    // Draw loss line
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < losses.length; i++) {
      const x = padding + (i / (losses.length - 1)) * plotWidth;
      const y = padding + (1 - (losses[i] - minLoss) / (maxLoss - minLoss)) * plotHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw accuracy line
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < accuracies.length; i++) {
      const x = padding + (i / (accuracies.length - 1)) * plotWidth;
      const y = padding + (1 - accuracies[i]) * plotHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw axes
    ctx.strokeStyle = "#c7b299";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Labels
    ctx.font = "10px IBM Plex Mono";
    ctx.fillStyle = "#5a4733";
    ctx.textAlign = "center";
    ctx.fillText("Steps", width / 2, height - 5);

  }, [history]);

  if (history.length < 2) {
    return (
      <div className="deep-chart empty">
        <p>Train to see loss/accuracy curves</p>
      </div>
    );
  }

  return (
    <div className="deep-chart">
      <div className="chart-legend">
        <span className="legend-loss">■ Loss</span>
        <span className="legend-acc">■ Accuracy</span>
      </div>
      <canvas ref={canvasRef} width={280} height={140} />
    </div>
  );
}

