import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Tooltip } from "./Tooltip";

const baseTooltip = { visible: true, text: "Hello", x: 10, y: 20 };

describe("Tooltip", () => {
  it("renders tooltip content when visible", () => {
    render(<Tooltip tooltip={baseTooltip} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });

  it("returns null when not visible", () => {
    const { container } = render(
      <Tooltip tooltip={{ ...baseTooltip, visible: false }} />,
    );
    expect(container.firstChild).toBeNull();
  });
});
