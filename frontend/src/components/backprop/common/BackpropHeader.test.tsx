import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { BackpropHeader } from "./BackpropHeader";

describe("BackpropHeader", () => {
  it("renders header copy", () => {
    render(<BackpropHeader />);
    expect(screen.getByText("Backprop Lab (Chapter 3)")).toBeInTheDocument();
  });
});
