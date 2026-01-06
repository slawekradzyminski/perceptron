import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { MlpControls } from "./MlpControls";

test("renders controls and triggers callbacks", () => {
  const onHiddenDimChange = vi.fn();
  const onLrChange = vi.fn();
  const onSeedChange = vi.fn();

  render(
    <MlpControls
      hiddenDim={2}
      lr={0.5}
      seed={0}
      loading={false}
      disabled={false}
      onHiddenDimChange={onHiddenDimChange}
      onLrChange={onLrChange}
      onSeedChange={onSeedChange}
    />,
  );

  fireEvent.change(screen.getByLabelText("Hidden units"), { target: { value: "3" } });
  fireEvent.change(screen.getByLabelText("Learning rate"), { target: { value: "0.25" } });
  fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "2" } });

  expect(onHiddenDimChange).toHaveBeenCalledWith(3);
  expect(onLrChange).toHaveBeenCalledWith(0.25);
  expect(onSeedChange).toHaveBeenCalledWith(2);
});
