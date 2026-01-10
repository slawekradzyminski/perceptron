import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TransformerInputSection } from "./TransformerInputSection";

describe("TransformerInputSection", () => {
  it("renders the input and handles changes", () => {
    const handleChange = vi.fn();
    render(
      <TransformerInputSection inputText="Hello" onInputChange={handleChange} />,
    );

    const textarea = screen.getByRole("textbox");
    expect(textarea).toHaveValue("Hello");

    fireEvent.change(textarea, { target: { value: "New text" } });
    expect(handleChange).toHaveBeenCalledWith("New text");
  });
});
