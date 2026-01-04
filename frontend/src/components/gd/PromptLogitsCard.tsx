import { useState } from "react";
import type { PromptLogitsResponse } from "../../types";

type PromptLogitsCardProps = {
  apiBase: string;
};

export function PromptLogitsCard({ apiBase }: PromptLogitsCardProps) {
  const [prompt, setPrompt] = useState("He walked into the room and");
  const [result, setResult] = useState<PromptLogitsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${apiBase}/gd/next-token-logprobs?prompt=${encodeURIComponent(prompt)}`,
      );
      if (!res.ok) throw new Error("Failed to load logits");
      const data = (await res.json()) as PromptLogitsResponse;
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load logits");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="lms-math lms-token-loss">
      <div className="lms-token-head">
        <div>
          <h3>Next-token logits</h3>
          <p>Inspect the live top candidates for the next token.</p>
        </div>
      </div>
      <form className="gd-prompt-form" onSubmit={onSubmit}>
        <label className="gd-prompt-label">
          Prompt
          <input
            type="text"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Enter a prompt..."
          />
        </label>
        <button type="submit" disabled={loading || prompt.trim().length === 0}>
          {loading ? "Loading..." : "Fetch logits"}
        </button>
      </form>
      {error && <p className="diag-error">{error}</p>}
      {result && (
        <>
          {result.warning && <p className="diag-note">{result.warning}</p>}
          <div className="lms-table-wrap lms-token-table">
            <table className="lms-table">
              <thead>
                <tr>
                  <th>Token</th>
                  <th>Probability</th>
                  <th>Logprob</th>
                </tr>
              </thead>
              <tbody>
                {result.tokens.map((token) => (
                  <tr key={`${token.token}-${token.rank}`}>
                    <td>{token.token || "(blank)"}</td>
                    <td>
                      <span
                        className="gd-tooltip"
                        data-tooltip={`prob = exp(logprob) = ${token.prob.toFixed(6)}`}
                      >
                        {token.prob.toFixed(6)}
                      </span>
                    </td>
                    <td>
                      <span
                        className="gd-tooltip"
                        data-tooltip={`logprob = ln(prob) = ${token.logprob.toFixed(6)}`}
                      >
                        {token.logprob.toFixed(6)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
