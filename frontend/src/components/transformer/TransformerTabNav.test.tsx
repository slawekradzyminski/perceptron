import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TransformerTabNav } from "./TransformerTabNav";

describe("TransformerTabNav", () => {
  it("renders tabs and handles selection", () => {
    const handleChange = vi.fn();
    render(<TransformerTabNav activeTab="tokenize" onTabChange={handleChange} />);

    expect(screen.getByRole("button", { name: "Tokenization" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Model Scale" }));
    expect(handleChange).toHaveBeenCalledWith("scale");
  });
});
