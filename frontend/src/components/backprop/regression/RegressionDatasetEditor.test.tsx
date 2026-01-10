import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { RegressionDatasetEditor } from "./RegressionDatasetEditor";

describe("RegressionDatasetEditor", () => {
  it("parses samples and applies dataset", () => {
    const handleApply = vi.fn();
    render(
      <RegressionDatasetEditor
        samples={[{ x: 1, y: 2 }]}
        onApply={handleApply}
      />,
    );

    const textarea = screen.getByRole("textbox");
    fireEvent.change(textarea, { target: { value: "1,2\n3,4" } });

    fireEvent.click(screen.getByRole("button", { name: "Apply Dataset" }));
    expect(handleApply).toHaveBeenCalledWith([
      { x: 1, y: 2 },
      { x: 3, y: 4 },
    ]);
  });
});
