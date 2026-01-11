import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { LogitLensTab } from "./LogitLensTab";
import type { LogitLensResult } from "../../hooks/transformer/useGlassBoxApi";

const mockLogitLens: LogitLensResult = {
  text: "Hello",
  tokens: ["Hello"],
  n_layers: 2,
  top_k: 3,
  model_name: "gpt2",
  layers: [
    {
      layer: 0,
      layer_name: "Embedding",
      predictions: [
        { token: " world", token_id: 995, probability: 0.15 },
        { token: " there", token_id: 612, probability: 0.08 },
        { token: ",", token_id: 11, probability: 0.05 },
      ],
    },
    {
      layer: 1,
      layer_name: "Layer 1",
      predictions: [
        { token: " world", token_id: 995, probability: 0.25 },
        { token: "!", token_id: 0, probability: 0.12 },
        { token: " there", token_id: 612, probability: 0.10 },
      ],
    },
    {
      layer: 2,
      layer_name: "Layer 2",
      predictions: [
        { token: " world", token_id: 995, probability: 0.45 },
        { token: "!", token_id: 0, probability: 0.20 },
        { token: " there", token_id: 612, probability: 0.08 },
      ],
    },
  ],
  explanation: "Logit Lens for gpt2",
};

describe("LogitLensTab", () => {
  it("renders placeholder when no data", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <LogitLensTab
        inputText="Hello"
        onInputChange={handleInputChange}
        loading={false}
        logitLens={null}
        onFetchLogitLens={handleFetch}
      />,
    );

    expect(screen.getByRole("button", { name: "Analyze Logit Lens" })).toBeInTheDocument();
    expect(
      screen.getByText(/internal monologue/),
    ).toBeInTheDocument();
  });

  it("triggers fetch on button click", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <LogitLensTab
        inputText="Hello"
        onInputChange={handleInputChange}
        loading={false}
        logitLens={null}
        onFetchLogitLens={handleFetch}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Analyze Logit Lens" }));
    expect(handleFetch).toHaveBeenCalledWith("Hello", 5);
  });

  it("renders logit lens results with table", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <LogitLensTab
        inputText="Hello"
        onInputChange={handleInputChange}
        loading={false}
        logitLens={mockLogitLens}
        onFetchLogitLens={handleFetch}
      />,
    );

    expect(screen.getByText("gpt2")).toBeInTheDocument();
    expect(screen.getByText("2 layers")).toBeInTheDocument();

    // Table headers
    expect(screen.getByText("Layer")).toBeInTheDocument();
    expect(screen.getByText("Top Prediction")).toBeInTheDocument();
    expect(screen.getByText("Confidence")).toBeInTheDocument();

    // Layer names
    expect(screen.getByText("Embedding")).toBeInTheDocument();
    expect(screen.getByText("Layer 1")).toBeInTheDocument();
    expect(screen.getByText("Layer 2")).toBeInTheDocument();
  });

  it("shows educational content", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <LogitLensTab
        inputText="Hello"
        onInputChange={handleInputChange}
        loading={false}
        logitLens={mockLogitLens}
        onFetchLogitLens={handleFetch}
      />,
    );

    expect(screen.getByText("What is the Logit Lens?")).toBeInTheDocument();
    expect(screen.getByText(/Early layers/)).toBeInTheDocument();
    expect(screen.getByText(/Later layers/)).toBeInTheDocument();
  });

  it("disables button when loading", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <LogitLensTab
        inputText="Hello"
        onInputChange={handleInputChange}
        loading={true}
        logitLens={null}
        onFetchLogitLens={handleFetch}
      />,
    );

    const button = screen.getByRole("button", { name: "Analyzing..." });
    expect(button).toBeDisabled();
  });

  it("allows text editing via textarea", () => {
    const handleFetch = vi.fn();
    const handleInputChange = vi.fn();
    render(
      <LogitLensTab
        inputText="Hello"
        onInputChange={handleInputChange}
        loading={false}
        logitLens={null}
        onFetchLogitLens={handleFetch}
      />,
    );

    const textarea = screen.getByRole("textbox");
    fireEvent.change(textarea, { target: { value: "New text" } });
    expect(handleInputChange).toHaveBeenCalledWith("New text");
  });
});
