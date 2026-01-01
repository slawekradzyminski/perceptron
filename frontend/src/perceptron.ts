export type Sample = { x: number[]; y: 1 | -1 };

export class Perceptron {
  w: number[];
  b: number;
  lr: number;

  constructor(dim: number, lr = 1) {
    if (dim <= 0) throw new Error("dim must be positive");
    this.w = Array.from({ length: dim }, () => 0);
    this.b = 0;
    this.lr = lr;
  }

  score(x: number[]): number {
    if (x.length !== this.w.length) throw new Error("x has wrong dimension");
    return x.reduce((acc, v, i) => acc + v * this.w[i], 0) + this.b;
  }

  predict(x: number[]): 1 | -1 {
    return this.score(x) >= 0 ? 1 : -1;
  }

  trainStep(x: number[], y: 1 | -1, lr = this.lr) {
    const s = this.score(x);
    const pred = s >= 0 ? 1 : -1;
    const mistake = y * s <= 0;
    if (mistake) {
      this.w = this.w.map((w, i) => w + lr * y * x[i]);
      this.b += lr * y;
    }
    return { score: s, pred, mistake, w: this.w, b: this.b };
  }
}

export const datasets: Record<string, Sample[]> = {
  or: [
    { x: [-1, -1], y: -1 },
    { x: [-1, 1], y: 1 },
    { x: [1, -1], y: 1 },
    { x: [1, 1], y: 1 },
  ],
  xor: [
    { x: [-1, -1], y: -1 },
    { x: [-1, 1], y: 1 },
    { x: [1, -1], y: 1 },
    { x: [1, 1], y: -1 },
  ],
};
