import { fireEvent } from "@testing-library/dom";
import { expect, test } from "vitest";
import { initApp } from "./app";

function setupDom() {
  document.body.innerHTML = `
    <div>
      <canvas id="plot" width="420" height="420"></canvas>
      <canvas id="input" width="200" height="120"></canvas>
      <canvas id="weights-before" width="200" height="120"></canvas>
      <canvas id="weights-after" width="200" height="120"></canvas>
      <canvas id="contrib" width="200" height="120"></canvas>
      <span id="score"></span>
      <input id="lr" type="range" min="0.1" max="2" step="0.1" value="1" />
      <span id="lr-value"></span>
      <select id="dataset"><option value="or">OR</option><option value="xor">XOR</option></select>
      <input id="api-base" type="text" value="http://127.0.0.1:8000" />
      <div id="calc-x"></div>
      <div id="calc-y"></div>
      <div id="calc-w"></div>
      <div id="calc-b"></div>
      <div id="calc-s"></div>
      <div id="calc-pred"></div>
      <div id="calc-score-eq"></div>
      <div id="calc-update-eq"></div>
      <div id="calc-note"></div>
      <div id="state-w"></div>
      <div id="state-b"></div>
      <div id="state-ds"></div>
      <div id="state-idx"></div>
      <div id="next-x"></div>
      <div id="next-y"></div>
      <div id="last-step-section"></div>
      <div id="cell-tooltip"></div>
      <div id="bias-before"></div>
      <div id="bias-after"></div>
      <button id="step">Step</button>
      <button id="reset">Reset</button>
    </div>
  `;
}

function mockCanvas() {
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
}

test("initApp wires buttons and updates score", async () => {
  mockCanvas();
  setupDom();
  let call = 0;
  global.fetch = async () => {
    call += 1;
    if (call === 1) {
      return {
        ok: true,
        json: async () => ({ w: [0, 0], b: 0, idx: 0, next_x: [-1, -1], next_y: -1 }),
      } as Response;
    }
    return {
      ok: true,
      json: async () => ({
        w: [1, 0],
        b: 1,
        idx: 1,
        x: [1, -1],
        y: 1,
        score: 1,
        pred: 1,
        mistake: false,
        delta_w: [0, 0],
        delta_b: 0,
        lr: 1,
        next_x: [-1, 1],
        next_y: 1,
      }),
    } as Response;
  };
  initApp(document);

  const score = document.getElementById("score") as HTMLElement;
  expect(score.textContent).not.toBe("");

  const initial = score.textContent;
  fireEvent.click(document.getElementById("step") as HTMLButtonElement);
  await new Promise((r) => setTimeout(r, 0));
  expect(score.textContent).not.toBe(initial);

  fireEvent.click(document.getElementById("reset") as HTMLButtonElement);
  await new Promise((r) => setTimeout(r, 0));
  expect(score.textContent).not.toBe("");
});
