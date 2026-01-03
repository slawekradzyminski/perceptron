import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { useHotkeys } from "./useHotkeys";

function HotkeyHarness({
  onStep,
  onReset,
  enabled = true,
}: {
  onStep: () => void;
  onReset: () => void;
  enabled?: boolean;
}) {
  useHotkeys({ onStep, onReset, enabled });
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

test("does not fire when disabled", () => {
  const onStep = vi.fn();
  const onReset = vi.fn();
  render(<HotkeyHarness onStep={onStep} onReset={onReset} enabled={false} />);
  fireEvent.keyDown(document, { key: "s" });
  fireEvent.keyDown(document, { key: "r" });
  expect(onStep).not.toHaveBeenCalled();
  expect(onReset).not.toHaveBeenCalled();
});
