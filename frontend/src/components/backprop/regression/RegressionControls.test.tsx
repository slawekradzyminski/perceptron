import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { RegressionControls } from "./RegressionControls";

describe("RegressionControls", () => {
  it("handles loss and learning rate changes", () => {
    const handleLossChange = vi.fn();
    const handleLrChange = vi.fn();
    const handleLrCommit = vi.fn();

    render(
      <RegressionControls
        loss="mse"
        lr={0.1}
        onLossChange={handleLossChange}
        onLrChange={handleLrChange}
        onLrCommit={handleLrCommit}
      />,
    );

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "l1" } });
    expect(handleLossChange).toHaveBeenCalledWith("l1");

    const slider = screen.getByRole("slider");
    fireEvent.change(slider, { target: { value: "0.5" } });
    expect(handleLrChange).toHaveBeenCalledWith(0.5);

    fireEvent.mouseUp(slider);
    expect(handleLrCommit).toHaveBeenCalled();
  });
});
