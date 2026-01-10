import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ConvEducation } from "./ConvEducation";

describe("ConvEducation", () => {
  it("renders main heading", () => {
    render(<ConvEducation />);

    expect(
      screen.getByText("How Convolutional Neural Networks Work")
    ).toBeInTheDocument();
  });

  it("renders convolution explanation section", () => {
    render(<ConvEducation />);

    expect(screen.getByText("🔍 What is a Convolution?")).toBeInTheDocument();
  });

  it("renders weight sharing section", () => {
    render(<ConvEducation />);

    expect(
      screen.getByText("🎯 Weight Sharing: The Key Insight")
    ).toBeInTheDocument();
  });

  it("renders padding and stride section", () => {
    render(<ConvEducation />);

    expect(screen.getByText("📐 Padding and Stride")).toBeInTheDocument();
  });

  it("renders feature hierarchies section", () => {
    render(<ConvEducation />);

    expect(screen.getByText("🏗️ Building Feature Hierarchies")).toBeInTheDocument();
  });

  it("renders math behind each cell section", () => {
    render(<ConvEducation />);

    expect(screen.getByText("🧮 The Math Behind Each Cell")).toBeInTheDocument();
  });

  it("contains important CNN concepts", () => {
    render(<ConvEducation />);

    // These terms appear multiple times, so use getAllByText
    expect(screen.getAllByText(/kernel/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/filter/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/activation map/i).length).toBeGreaterThan(0);
  });

  it("explains layer hierarchy", () => {
    render(<ConvEducation />);

    expect(screen.getByText(/Layer 1/)).toBeInTheDocument();
    expect(screen.getByText(/Layer 2/)).toBeInTheDocument();
    expect(screen.getByText(/Layer 3/)).toBeInTheDocument();
  });

  it("explains weight sharing benefits", () => {
    render(<ConvEducation />);

    expect(screen.getByText(/Fewer parameters/)).toBeInTheDocument();
    expect(screen.getByText(/Translation invariance/)).toBeInTheDocument();
    expect(screen.getByText(/Efficient learning/)).toBeInTheDocument();
  });

  it("explains calculation steps", () => {
    render(<ConvEducation />);

    expect(screen.getByText(/Extract patch/)).toBeInTheDocument();
    expect(screen.getByText(/Multiply/)).toBeInTheDocument();
    expect(screen.getByText(/Sum/)).toBeInTheDocument();
  });
});
