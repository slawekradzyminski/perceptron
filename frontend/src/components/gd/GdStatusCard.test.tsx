import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { GdStatusCard } from "./GdStatusCard";

test("renders status info when available", () => {
  render(
    <GdStatusCard
      status={{
        ok: true,
        model: "llama3.2:1b",
        base_url: "http://127.0.0.1:11434",
        last_latency_ms: 12.5,
        models: ["llama3.2:1b"],
      }}
    />,
  );
  expect(screen.getByText("Ollama status")).toBeInTheDocument();
  expect(screen.getByText("llama3.2:1b")).toBeInTheDocument();
});
