import { useMemo, useState } from "react";

type ComputedRow = {
  context: string;
  correctToken: string;
  pCorrect: number;
  topToken: string;
  topProb: number;
  ceLoss: number;
  l1Loss: number;
};

type TokenLossRow = {
  context: string;
  correct_token: string;
  p_correct: number;
  top_token: string;
  top_prob: number;
  l1_loss: number;
  ce_loss: number;
};

type TokenLossExample = {
  id: string;
  title: string;
  description: string;
  average: { l1: number; ce: number };
  rows: TokenLossRow[];
};

function formatProb(value: number) {
  if (value === 0) return "0";
  // Always use decimal notation for consistency (up to 6 decimal places)
  return value.toFixed(6);
}

type TokenLossTableCardProps = {
  apiBase: string;
  examples: TokenLossExample[];
};

export function TokenLossTableCard({ apiBase, examples }: TokenLossTableCardProps) {
  const [localExamples, setLocalExamples] = useState(examples);
  const [exampleId, setExampleId] = useState(examples[0]?.id ?? "");
  const [customPrompt, setCustomPrompt] = useState("");
  const [customError, setCustomError] = useState<string | null>(null);
  const [customLoading, setCustomLoading] = useState(false);
  const example = useMemo(
    () => localExamples.find((item) => item.id === exampleId) ?? localExamples[0],
    [localExamples, exampleId],
  );

  const rows = useMemo<ComputedRow[]>(
    () => {
      if (!example) return [];
      return example.rows.map((row) => ({
        context: row.context,
        correctToken: row.correct_token,
        pCorrect: row.p_correct,
        topToken: row.top_token,
        topProb: row.top_prob,
        ceLoss: row.ce_loss,
        l1Loss: row.l1_loss,
      }));
    },
    [example],
  );

  const { minLoss, maxLoss, avgCeLoss, avgL1Loss } = useMemo(() => {
    if (rows.length === 0) return { minLoss: 0, maxLoss: 0, avgCeLoss: 0, avgL1Loss: 0 };
    const ceLosses = rows.map((row) => row.ceLoss);
    return {
      minLoss: Math.min(...ceLosses),
      maxLoss: Math.max(...ceLosses),
      avgCeLoss: ceLosses.reduce((acc, val) => acc + val, 0) / ceLosses.length,
      avgL1Loss: rows.reduce((acc, row) => acc + row.l1Loss, 0) / rows.length,
    };
  }, [rows]);

  if (!example) {
    return (
      <div className="lms-math lms-token-loss">
        <h3>Token loss table</h3>
        <p className="lms-empty">No token loss examples available.</p>
      </div>
    );
  }

  return (
    <div className="lms-math lms-token-loss">
      <div className="lms-token-head">
        <div>
          <h3>Token loss table</h3>
        </div>
        <label className="lms-token-select">
          Example
          <select value={exampleId} onChange={(event) => setExampleId(event.target.value)}>
            {localExamples.map((item) => (
              <option key={item.id} value={item.id}>
                {item.title}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="gd-prompt-form">
        <label className="gd-prompt-label">
          Custom prompt
          <input
            type="text"
            value={customPrompt}
            onChange={(event) => setCustomPrompt(event.target.value)}
            placeholder="Enter a custom prompt..."
          />
        </label>
        <button
          type="button"
          disabled={customLoading || customPrompt.trim().length === 0}
          onClick={async () => {
            setCustomLoading(true);
            setCustomError(null);
            try {
              const res = await fetch(
                `${apiBase}/gd/token-losses?source=ollama&example_id=custom&prompt=${encodeURIComponent(customPrompt)}`,
              );
              if (!res.ok) throw new Error("Failed to load custom prompt");
              const data = (await res.json()) as { examples: TokenLossExample[] };
              const customExample = data.examples?.[0];
              if (customExample) {
                setLocalExamples((prev) => {
                  const next = prev.filter((item) => item.id !== "custom");
                  return [...next, customExample];
                });
                setExampleId("custom");
              }
            } catch (err) {
              setCustomError(err instanceof Error ? err.message : "Failed to load custom prompt");
            } finally {
              setCustomLoading(false);
            }
          }}
        >
          {customLoading ? "Loading..." : "Load"}
        </button>
      </div>
      {customError && <p className="diag-error">{customError}</p>}
      <div className="lms-token-controls" role="group" aria-label="Loss summary">
        <div className="lms-token-average">
          Average cross-entropy: <strong>{avgCeLoss.toFixed(4)}</strong>
        </div>
        <div className="lms-token-average">
          Average L1: <strong>{avgL1Loss.toFixed(4)}</strong>
        </div>
      </div>
      <div className="lms-table-wrap lms-token-table">
        <table className="lms-table">
          <thead>
            <tr>
              <th>Input text</th>
              <th>Top token</th>
              <th>P(top)</th>
              <th>Correct token</th>
              <th>P(correct)</th>
              <th>CE loss</th>
              <th>L1 loss</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isMin = row.ceLoss === minLoss;
              const isMax = row.ceLoss === maxLoss;
              return (
                <tr key={`${row.context}-${row.correctToken}`}>
                  <td>{row.context}</td>
                  <td>{row.topToken || "(n/a)"}</td>
                  <td>{row.topProb ? formatProb(row.topProb) : "-"}</td>
                  <td>{row.correctToken}</td>
                  <td>{formatProb(row.pCorrect)}</td>
                  <td className={isMax ? "bad" : isMin ? "ok" : ""}>
                    <span
                      className="gd-tooltip"
                      data-tooltip={`CE = -ln(p) = -ln(${row.pCorrect.toFixed(6)})`}
                    >
                      {row.ceLoss.toFixed(4)}
                    </span>
                  </td>
                  <td>
                    <span
                      className="gd-tooltip"
                      data-tooltip={`L1 = 1 - p = 1 - ${row.pCorrect.toFixed(6)}`}
                    >
                      {row.l1Loss.toFixed(4)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="lms-note">Highest loss highlights the most confident mistakes; lowest loss shows the most confident hits.</p>
    </div>
  );
}
