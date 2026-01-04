import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { TokenLossTableCard } from "./TokenLossTableCard";

test("renders token loss table and toggles metric", () => {
  const examples = [
    {
      id: "example",
      title: "Example sentence",
      description: "Example description.",
      average: { l1: 0.5, ce: 1.0 },
      rows: [
        {
          context: "<begin_of_text>",
          correct_token: "The",
          p_correct: 0.5,
          top_token: "The",
          top_prob: 0.5,
          l1_loss: 0.5,
          ce_loss: 0.6931,
        },
      ],
    },
  ];
  render(<TokenLossTableCard examples={examples} />);
  expect(screen.getByText("Token loss table")).toBeInTheDocument();
  expect(screen.getByLabelText("Loss metric")).toBeInTheDocument();
  expect(screen.getByText("Average cross-entropy:")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "L1 loss" }));
  expect(screen.getByText("Average l1:")).toBeInTheDocument();
});
