import { useState, useRef, useEffect } from "react";

type DeepNetworkPreviewProps = {
  inputDim: number;
  hiddenDims: number[];
  outputDim: number;
};

// Sizes for clear visibility
const NODE_RADIUS = 18;
const LAYER_GAP = 100;
const NODE_GAP = 50;
const MAX_NODES_DISPLAY = 8;
const CANVAS_PADDING = 50;

// Colors matching project aesthetic
const COLORS = {
  input: { fill: "#dbeafe", stroke: "#3b82f6" },
  hidden: { fill: "#dcfce7", stroke: "#22c55e" },
  output: { fill: "#fef3c7", stroke: "#f59e0b" },
  connection: "#c7b299",
  text: "#4a3b2b",
};

function drawNetwork(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  inputDim: number,
  hiddenDims: number[],
  outputDim: number
) {
  ctx.clearRect(0, 0, width, height);

  const layers = [inputDim, ...hiddenDims, outputDim];
  const numLayers = layers.length;

  // Calculate layer positions
  const layerX = layers.map((_, i) => 
    CANVAS_PADDING + (i * (width - 2 * CANVAS_PADDING)) / (numLayers - 1)
  );

  // Calculate node positions for each layer
  const nodePositions: Array<Array<{ x: number; y: number }>> = [];
  
  for (let l = 0; l < numLayers; l++) {
    const nodesInLayer = Math.min(layers[l], MAX_NODES_DISPLAY);
    const totalHeight = (nodesInLayer - 1) * NODE_GAP;
    const startY = (height - totalHeight) / 2;
    
    const positions: Array<{ x: number; y: number }> = [];
    for (let n = 0; n < nodesInLayer; n++) {
      positions.push({
        x: layerX[l],
        y: startY + n * NODE_GAP,
      });
    }
    
    nodePositions.push(positions);
  }

  // Draw connections
  ctx.strokeStyle = COLORS.connection;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.4;

  for (let l = 0; l < numLayers - 1; l++) {
    const currentLayer = nodePositions[l];
    const nextLayer = nodePositions[l + 1];

    for (const from of currentLayer) {
      for (const to of nextLayer) {
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      }
    }
  }

  ctx.globalAlpha = 1.0;

  // Draw nodes
  for (let l = 0; l < numLayers; l++) {
    const isInput = l === 0;
    const isOutput = l === numLayers - 1;
    const color = isInput ? COLORS.input : isOutput ? COLORS.output : COLORS.hidden;

    const nodesInLayer = Math.min(layers[l], MAX_NODES_DISPLAY);
    const hasMore = layers[l] > MAX_NODES_DISPLAY;

    for (let n = 0; n < nodesInLayer; n++) {
      const pos = nodePositions[l][n];
      
      // Draw node
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, NODE_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = color.fill;
      ctx.fill();
      ctx.strokeStyle = color.stroke;
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    // Draw "+N more" if truncated
    if (hasMore) {
      ctx.fillStyle = "#8a7562";
      ctx.font = "12px IBM Plex Mono";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const lastPos = nodePositions[l][nodesInLayer - 1];
      ctx.fillText(`+${layers[l] - MAX_NODES_DISPLAY}`, layerX[l], lastPos.y + NODE_GAP * 0.7);
    }
  }

  // Draw layer labels at top
  ctx.textBaseline = "top";
  
  const labels = ["Input", ...hiddenDims.map((_, i) => `H${i + 1}`), "Output"];
  const counts = [inputDim, ...hiddenDims, outputDim];

  for (let l = 0; l < numLayers; l++) {
    const isInput = l === 0;
    const isOutput = l === numLayers - 1;
    const labelColor = isInput ? COLORS.input.stroke : isOutput ? COLORS.output.stroke : COLORS.hidden.stroke;
    
    ctx.font = "bold 13px Space Grotesk";
    ctx.textAlign = "center";
    ctx.fillStyle = COLORS.text;
    ctx.fillText(labels[l], layerX[l], 10);
    
    ctx.font = "bold 14px IBM Plex Mono";
    ctx.fillStyle = labelColor;
    ctx.fillText(`${counts[l]}`, layerX[l], 28);
  }
}

export function DeepNetworkPreview({
  inputDim,
  hiddenDims,
  outputDim,
}: DeepNetworkPreviewProps) {
  const [isHovered, setIsHovered] = useState(false);
  const triggerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const depth = hiddenDims.length;
  const width = hiddenDims[0] || 8;
  
  // Find max neurons in any layer for height calculation
  const maxNeurons = Math.min(MAX_NODES_DISPLAY, Math.max(inputDim, ...hiddenDims, outputDim));

  // Calculate canvas size
  const canvasWidth = Math.max(400, (depth + 2) * LAYER_GAP + 2 * CANVAS_PADDING);
  const canvasHeight = Math.max(300, maxNeurons * NODE_GAP + 2 * CANVAS_PADDING);

  useEffect(() => {
    if (isHovered && canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) {
        drawNetwork(ctx, canvasWidth, canvasHeight, inputDim, hiddenDims, outputDim);
      }
    }
  }, [isHovered, inputDim, hiddenDims, outputDim, canvasWidth, canvasHeight]);

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
  };

  // Calculate total params
  let params = inputDim * hiddenDims[0] + hiddenDims[0]; // First hidden layer
  for (let i = 1; i < hiddenDims.length; i++) {
    params += hiddenDims[i - 1] * hiddenDims[i] + hiddenDims[i];
  }
  params += hiddenDims[hiddenDims.length - 1] * outputDim + outputDim; // Output layer

  return (
    <div className="arch-preview-wrapper">
      <div
        ref={triggerRef}
        className="arch-preview arch-preview-hoverable"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        <span>{inputDim} → [{hiddenDims.join(" × ")}] → {outputDim}</span>
        <span className="hover-hint">hover for diagram</span>
      </div>

      {isHovered && (
        <div className="network-popup-inline">
          <div className="network-popup-header">
            Network Architecture
          </div>
          <canvas
            ref={canvasRef}
            width={canvasWidth}
            height={canvasHeight}
            className="network-popup-canvas"
          />
          <div className="network-popup-footer">
            {depth} layer{depth !== 1 ? "s" : ""} × {width} neurons = {params} params
          </div>
        </div>
      )}
    </div>
  );
}
