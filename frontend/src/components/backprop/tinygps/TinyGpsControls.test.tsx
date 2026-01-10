import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TinyGpsControls } from "./TinyGpsControls";

describe("TinyGpsControls", () => {
  it("renders controls and handles interactions", () => {
    const handleDatasetChange = vi.fn();
    const handleLrChange = vi.fn();
    const handleLrCommit = vi.fn();

    render(
      <TinyGpsControls
        datasets={["europe", "usa"]}
        dataset="europe"
        lr={0.2}
        coords={[["Paris", [[48.1, 2.3]]]]}
        onDatasetChange={handleDatasetChange}
        onLrChange={handleLrChange}
        onLrCommit={handleLrCommit}
      />,
    );

    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "usa" },
    });
    expect(handleDatasetChange).toHaveBeenCalledWith("usa");

    const slider = screen.getByRole("slider");
    fireEvent.change(slider, { target: { value: "0.5" } });
    expect(handleLrChange).toHaveBeenCalledWith(0.5);

    fireEvent.mouseUp(slider);
    expect(handleLrCommit).toHaveBeenCalled();

    expect(screen.getByText("Paris")).toBeInTheDocument();
  });
});
