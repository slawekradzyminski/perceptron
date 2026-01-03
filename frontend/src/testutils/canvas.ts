import { vi } from "vitest";

export function mockCanvas() {
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    canvas: { width: 100, height: 50 },
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    strokeRect: vi.fn(),
    setLineDash: vi.fn(),
    set fillStyle(_: string) {},
    set strokeStyle(_: string) {},
    set lineWidth(_: number) {},
  })) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}
