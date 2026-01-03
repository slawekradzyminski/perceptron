import { render, screen, waitFor } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { MlpTrainerPage } from "./MlpTrainerPage";
import { mockCanvas } from "../testutils/canvas";
import { mockMlpReset } from "../testutils/mocks/apiResponses";

test("renders MLP trainer page", async () => {
  mockCanvas();
  const fetchMock = vi.fn(async (url: RequestInfo) => {
    const target = typeof url === "string" ? url : url.url;
    if (target.endsWith("/mlp/reset")) {
      return {
        ok: true,
        json: async () => mockMlpReset(),
      } as Response;
    }
    throw new Error("unexpected");
  }) as typeof fetch;
  global.fetch = fetchMock;

  render(
    <MlpTrainerPage
      apiBase="http://127.0.0.1:8000"
      datasetName="xor"
      customConfig={{ rows: 1, cols: 2, samples: [] }}
      customApplied={true}
    />,
  );

  await waitFor(() => expect(screen.getByText("MLP (2-layer) training")).toBeInTheDocument());
});
