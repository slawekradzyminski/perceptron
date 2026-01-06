import { useEffect, useRef } from "react";
import type { DeepBoundary, DeepSample } from "../../hooks/deep/useDeepApi";

type DeepBoundaryCanvasProps = {
  boundary: DeepBoundary | null;
  samples: DeepSample[];
  showTiling: boolean;
  dataset?: string;
};

// Generate a deterministic color for a region ID
function regionColor(id: number): string {
  // Use golden ratio for nice color distribution
  const hue = (id * 137.508) % 360;
  const sat = 55 + (id % 3) * 15;
  const light = 50 + (id % 5) * 8;
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

// Baarle-Hertog enclave centers (normalized to [-1, 1] - matching backend datasets.py)
const BAARLE_ENCLAVES: Array<{ cx: number; cy: number; radius: number }> = [
  // Main Belgian area (H1) - large irregular shape in bottom-left/center
  { cx: -0.3, cy: 0.4, radius: 0.35 },
  // H2 - bottom right
  { cx: 0.5, cy: 0.5, radius: 0.12 },
  // H3 - right side
  { cx: 0.45, cy: 0.25, radius: 0.08 },
  // H4 - right side lower
  { cx: 0.55, cy: 0.35, radius: 0.07 },
  // H5 - right side center
  { cx: 0.5, cy: 0.1, radius: 0.06 },
  // H6 - left upper area
  { cx: -0.35, cy: -0.2, radius: 0.08 },
  // H7 - far left
  { cx: -0.7, cy: 0.0, radius: 0.06 },
  // H8 - upper center
  { cx: 0.2, cy: -0.3, radius: 0.15 },
  // H9 - upper right
  { cx: 0.55, cy: -0.15, radius: 0.06 },
  // H10 - far upper right
  { cx: 0.75, cy: -0.55, radius: 0.05 },
  // H11 - upper right
  { cx: 0.5, cy: -0.45, radius: 0.06 },
  // H12 - upper center-right
  { cx: 0.35, cy: -0.55, radius: 0.04 },
  // H13 - upper left
  { cx: 0.0, cy: -0.4, radius: 0.05 },
  // H14 - left side
  { cx: -0.2, cy: -0.35, radius: 0.05 },
  // H15 - center left
  { cx: -0.1, cy: -0.15, radius: 0.08 },
  // H16 - right side
  { cx: 0.7, cy: -0.2, radius: 0.06 },
];

// Draw the Baarle-Hertog map background
function drawBaarleMap(ctx: CanvasRenderingContext2D, size: number): void {
  // Fill with Netherlands color (light blue - representing water/land)
  ctx.fillStyle = "#dbeafe"; // Light blue for Netherlands
  ctx.fillRect(0, 0, size, size);

  // Draw a subtle grid pattern for the map aesthetic
  ctx.strokeStyle = "#bfdbfe";
  ctx.lineWidth = 0.5;
  const gridStep = size / 20;
  for (let i = 0; i <= 20; i++) {
    ctx.beginPath();
    ctx.moveTo(i * gridStep, 0);
    ctx.lineTo(i * gridStep, size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * gridStep);
    ctx.lineTo(size, i * gridStep);
    ctx.stroke();
  }

  // Draw Belgium enclaves (yellow - Belgian national color)
  ctx.fillStyle = "#fbbf24"; // Golden yellow for Belgium
  ctx.strokeStyle = "#92400e"; // Brown border
  ctx.lineWidth = 1.5;

  for (const enclave of BAARLE_ENCLAVES) {
    // Convert from [-1, 1] to canvas coordinates
    const px = ((enclave.cx + 1) / 2) * size;
    const py = ((1 - (enclave.cy + 1) / 2)) * size; // Flip Y
    const pr = enclave.radius * size / 2;

    // Draw as slightly irregular shapes for realism
    ctx.beginPath();

    // Create an irregular polygon approximating a circle
    const points = 12;
    for (let i = 0; i <= points; i++) {
      const angle = (i / points) * Math.PI * 2;
      const variance = 0.85 + Math.sin(angle * 3 + enclave.cx * 5) * 0.15;
      const rx = px + pr * Math.cos(angle) * variance;
      const ry = py + pr * Math.sin(angle) * variance;
      if (i === 0) {
        ctx.moveTo(rx, ry);
      } else {
        ctx.lineTo(rx, ry);
      }
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // Add some "roads" as gray lines
  ctx.strokeStyle = "#9ca3af";
  ctx.lineWidth = 1;
  // Main roads crossing the territory
  ctx.beginPath();
  ctx.moveTo(0, size * 0.3);
  ctx.lineTo(size, size * 0.35);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(size * 0.2, 0);
  ctx.lineTo(size * 0.25, size);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(size * 0.6, 0);
  ctx.lineTo(size * 0.55, size);
  ctx.stroke();
}

// Class colors - Belgium (yellow) and Netherlands (blue)
const BAARLE_COLORS = {
  belgium: "#fbbf24",
  netherlands: "#60a5fa",
};

// Default class colors for other datasets
const CLASS_COLORS = ["#3b82f6", "#ef4444"]; // Blue for class 0, Red for class 1

export function DeepBoundaryCanvas({
  boundary,
  samples,
  showTiling,
  dataset = "circles",
}: DeepBoundaryCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isBaarle = dataset === "baarle";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const size = canvas.width;

    // Clear canvas
    ctx.clearRect(0, 0, size, size);

    // For Baarle dataset, draw the map background first
    if (isBaarle) {
      drawBaarleMap(ctx, size);
    } else {
      // Default background
      ctx.fillStyle = "#fffdf8";
      ctx.fillRect(0, 0, size, size);
    }

    // Draw boundary visualization if available
    if (boundary) {
      const { resolution, predictions, region_ids } = boundary;
      const cellSize = size / resolution;

      // Draw grid (note: j=0 in backend is y=-1, which should be at bottom of canvas)
      for (let j = 0; j < resolution; j++) {
        for (let i = 0; i < resolution; i++) {
          const x = i * cellSize;
          // Flip Y: backend j=0 is y=-1 (bottom), canvas y=0 is top
          const y = (resolution - 1 - j) * cellSize;

          if (showTiling) {
            // Tiling view: color by region ID
            const regionId = region_ids[j][i];
            ctx.fillStyle = regionColor(regionId);
            ctx.globalAlpha = 0.7;
          } else {
            // Boundary view: color by predicted class
            const pred = predictions[j][i];
            if (isBaarle) {
              // Use Belgium/Netherlands colors
              ctx.fillStyle = pred === 1 ? BAARLE_COLORS.belgium : BAARLE_COLORS.netherlands;
            } else {
              ctx.fillStyle = CLASS_COLORS[pred];
            }
            ctx.globalAlpha = 0.35;
          }

          ctx.fillRect(x, y, cellSize + 1, cellSize + 1);
        }
      }
      ctx.globalAlpha = 1.0;
    }

    // Draw sample points
    for (const sample of samples) {
      // Convert from [-1, 1] to canvas coordinates
      const px = ((sample.x[0] + 1) / 2) * size;
      const py = ((1 - (sample.x[1] + 1) / 2)) * size; // Flip Y

      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);

      if (isBaarle) {
        // Belgium = yellow (class 1), Netherlands = blue (class 0)
        ctx.fillStyle = sample.y === 1 ? BAARLE_COLORS.belgium : BAARLE_COLORS.netherlands;
        ctx.strokeStyle = sample.y === 1 ? "#92400e" : "#1e40af";
      } else {
        ctx.fillStyle = CLASS_COLORS[sample.y];
        ctx.strokeStyle = "#fff";
      }

      ctx.fill();
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [boundary, samples, showTiling, isBaarle]);

  return (
    <div className="deep-boundary">
      <canvas
        ref={canvasRef}
        width={400}
        height={400}
        className="boundary-canvas"
      />
      <div className="boundary-legend">
        {isBaarle ? (
          <>
            <span className="legend-item">
              <span className="legend-dot belgium" />
              Belgium (Baarle-Hertog)
            </span>
            <span className="legend-item">
              <span className="legend-dot netherlands" />
              Netherlands (Baarle-Nassau)
            </span>
          </>
        ) : (
          <>
            <span className="legend-item">
              <span className="legend-dot class-0" />
              Class 0
            </span>
            <span className="legend-item">
              <span className="legend-dot class-1" />
              Class 1
            </span>
          </>
        )}
        {showTiling && (
          <span className="legend-note">Colors = Linear Regions</span>
        )}
      </div>
    </div>
  );
}
