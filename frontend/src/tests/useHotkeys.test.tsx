import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { useHotkeys } from "../hooks/useHotkeys";

function HotkeyHarness({ onStep, onReset }: { onStep: () => void; onReset: () => void }) {
  useHotkeys({ onStep, onReset });
  return <input aria-label="input" />;
}

test("fires callbacks on s/r keys", () => {
  const onStep = vi.fn();
  const onReset = vi.fn();
  render(<HotkeyHarness onStep={onStep} onReset={onReset} />);

  fireEvent.keyDown(document, { key: "s" });
  fireEvent.keyDown(document, { key: "r" });
  expect(onStep).toHaveBeenCalled();
  expect(onReset).toHaveBeenCalled();
});

test("does not fire when typing in inputs", () => {
  const onStep = vi.fn();
  const onReset = vi.fn();
  render(<HotkeyHarness onStep={onStep} onReset={onReset} />);
  const input = screen.getByLabelText("input");
  fireEvent.keyDown(input, { key: "s" });
  expect(onStep).not.toHaveBeenCalled();
});
