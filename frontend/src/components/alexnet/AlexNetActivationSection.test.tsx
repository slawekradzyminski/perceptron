import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { AlexNetActivationSection } from "./AlexNetActivationSection";
import type { ActivationGrid } from "../../hooks/alexnet/useAlexNetApi";

const mockActivations: ActivationGrid = {
  rows: 1,
  cols: 2,
  count: 2,
  layer: 1,
  layer_info: { name: "Conv1", filters: 64, kernel_size: 11, in_channels: 3 },
  sample_name: "gradient",
  input_image: "dGVzdA==",
  activations: [
    { index: 0, width: 55, height: 55, max_activation: 1.2, image: "dGVzdA==" },
    { index: 1, width: 55, height: 55, max_activation: 0.9, image: "dGVzdA==" },
  ],
};

describe("AlexNetActivationSection", () => {
  it("renders activation images", () => {
    const { container } = render(
      <AlexNetActivationSection
        activations={mockActivations}
        selectedSample="gradient"
        customImageUrl={null}
        hoveredFilter={null}
      />,
    );

    const activationImages = container.querySelectorAll(".activation-cell img");
    expect(activationImages.length).toBe(2);
  });
});
