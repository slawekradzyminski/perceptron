import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { LmsTooltip } from "./LmsTooltip";

test("renders tooltip content", () => {
  render(
    <LmsTooltip tooltip={{ visible: true, text: "Example", x: 10, y: 10 }} />,
  );
  expect(screen.getByText("Example")).toBeInTheDocument();
});
