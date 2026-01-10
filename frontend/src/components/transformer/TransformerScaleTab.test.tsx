import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TransformerScaleTab } from "./TransformerScaleTab";
import type { GrowthData, ScaleModel } from "../../hooks/transformer/useTransformerApi";

const mockModels: ScaleModel[] = [
  {
    name: "AlexNet",
    year: 2012,
    params: 61000000,
    params_formatted: "61.0M",
    type: "CNN",
    description: "ImageNet",
    input_size: "224×224",
    notable: "ReLU",
  },
  {
    name: "GPT-4",
    year: 2023,
    params: 1800000000000,
    params_formatted: "1.8T",
    type: "Transformer",
    description: "LLM",
    input_size: "32k tokens",
    notable: "Scale",
  },
];

const mockGrowth: GrowthData = {
  data: [
    { name: "AlexNet", year: 2012, params: 61000000, log_params: 8, type: "CNN" },
    { name: "GPT-4", year: 2023, params: 1800000000000, log_params: 12, type: "Transformer" },
  ],
  insight: "Growth insight",
};

describe("TransformerScaleTab", () => {
  it("renders scale controls and compares models", async () => {
    const handleCompare = vi.fn();
    render(
      <TransformerScaleTab
        scaleModels={mockModels}
        comparison={null}
        growth={mockGrowth}
        loading={false}
        onCompare={handleCompare}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("All Models")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(handleCompare).toHaveBeenCalledWith("AlexNet", "GPT-4");
  });
});
