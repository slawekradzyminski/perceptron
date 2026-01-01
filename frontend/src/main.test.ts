import { fireEvent } from "@testing-library/dom";
import { expect, test } from "vitest";
import { initApp } from "./app";

function setupDom() {
  document.body.innerHTML = `
    <div>
      <canvas id="plot" width="420" height="420"></canvas>
      <canvas id="input" width="200" height="200"></canvas>
      <canvas id="weights" width="200" height="200"></canvas>
      <canvas id="contrib" width="200" height="200"></canvas>
      <span id="score"></span>
      <input id="lr" type="range" min="0.1" max="2" step="0.1" value="1" />
      <span id="lr-value"></span>
      <select id="dataset"><option value="or">OR</option><option value="xor">XOR</option></select>
      <select id="grid-size"><option value="2">2</option><option value="3" selected>3</option></select>
      <input id="api-base" type="text" value="http://127.0.0.1:8000" />
      <button id="step">Step</button>
      <button id="reset">Reset</button>
    </div>
  `;
}

function mockCanvas() {
  HTMLCanvasElement.prototype.getContext = () =>
    ({
      canvas: { width: 100, height: 100 },
      clearRect: () => undefined,
      fillRect: () => undefined,
      beginPath: () => undefined,
      moveTo: () => undefined,
      lineTo: () => undefined,
      stroke: () => undefined,
      arc: () => undefined,
      fill: () => undefined,
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
        json: async () => ({ w: [1, 0], b: 1, idx: 1 }),
      } as Response;
    }
    return {
      ok: true,
      json: async () => ({ w: [2, 0], b: 2, idx: 2 }),
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
