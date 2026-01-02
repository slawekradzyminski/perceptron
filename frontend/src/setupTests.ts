import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

HTMLCanvasElement.prototype.getContext = () =>
  ({
    canvas: { width: 100, height: 50 },
    clearRect: () => undefined,
    fillRect: () => undefined,
    beginPath: () => undefined,
    moveTo: () => undefined,
    lineTo: () => undefined,
    stroke: () => undefined,
    arc: () => undefined,
    fill: () => undefined,
    strokeRect: () => undefined,
    setLineDash: () => undefined,
    set fillStyle(_: string) {},
    set strokeStyle(_: string) {},
    set lineWidth(_: number) {},
  }) as unknown as CanvasRenderingContext2D;
