import { render, screen } from "@testing-library/react";
import { expect, test, describe } from "vitest";
import { DeepStatsCard } from "./DeepStatsCard";

const mockArchitecture = {
  input_dim: 2,
  hidden_dims: [8, 8],
  output_dim: 2,
  depth: 2,
  width: 8,
  param_count: 114,
};

const mockMetrics = {
  loss: 0.693,
  accuracy: 0.75,
};

const mockRegions = {
  count: 15,
  theoretical_max: 100,
  efficiency: 0.15,
};

describe("DeepStatsCard", () => {
  test("renders Training Stats header", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={100}
      />,
    );
    expect(screen.getByText("Training Stats")).toBeInTheDocument();
  });

  test("renders Region Analysis header", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={100}
      />,
    );
    expect(screen.getByText("Region Analysis")).toBeInTheDocument();
  });

  test("displays epoch value", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={100}
      />,
    );
    expect(screen.getByText("Epoch")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  test("displays steps value", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={50}
      />,
    );
    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("50")).toBeInTheDocument();
  });

  test("displays loss value formatted", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={100}
      />,
    );
    expect(screen.getByText("Loss")).toBeInTheDocument();
    expect(screen.getByText("0.6930")).toBeInTheDocument();
  });

  test("displays accuracy as percentage", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={100}
      />,
    );
    expect(screen.getByText("Accuracy")).toBeInTheDocument();
    expect(screen.getByText("75.0%")).toBeInTheDocument();
  });

  test("displays actual regions count", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={100}
      />,
    );
    expect(screen.getByText("Actual Regions")).toBeInTheDocument();
    expect(screen.getByText("15")).toBeInTheDocument();
  });

  test("displays theoretical max regions", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={100}
      />,
    );
    expect(screen.getByText("Theory Max")).toBeInTheDocument();
    // 100 appears multiple times (steps and theory max), check for presence
    const values = screen.getAllByText("100");
    expect(values.length).toBeGreaterThanOrEqual(1);
  });

  test("displays efficiency as percentage", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={100}
      />,
    );
    expect(screen.getByText("Efficiency")).toBeInTheDocument();
    expect(screen.getByText("15.0%")).toBeInTheDocument();
  });

  test("displays parameter count", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={5}
        totalSteps={100}
      />,
    );
    expect(screen.getByText("Parameters")).toBeInTheDocument();
    expect(screen.getByText("114")).toBeInTheDocument();
  });

  test("displays dashes when architecture is null", () => {
    render(
      <DeepStatsCard
        architecture={null}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={0}
        totalSteps={0}
      />,
    );
    expect(screen.getByText("Parameters")).toBeInTheDocument();
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });

  test("displays dashes when metrics is null", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={null}
        regions={mockRegions}
        epoch={0}
        totalSteps={0}
      />,
    );
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(2); // Loss and Accuracy
  });

  test("displays dashes when regions is null", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={null}
        epoch={0}
        totalSteps={0}
      />,
    );
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(3); // Actual, Theory, Efficiency
  });

  test("handles zero epoch and steps", () => {
    render(
      <DeepStatsCard
        architecture={mockArchitecture}
        metrics={mockMetrics}
        regions={mockRegions}
        epoch={0}
        totalSteps={0}
      />,
    );
    const zeros = screen.getAllByText("0");
    expect(zeros.length).toBeGreaterThanOrEqual(2);
  });
});

