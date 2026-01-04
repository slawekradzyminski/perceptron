import { useMemo, useState } from "react";
import type { TinyGpsState } from "../../types";

function parseNumberList(value: string) {
  if (!value.trim()) return [] as number[];
  return value
    .split(/[,\\n]/)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => Number(chunk));
}

function parseOrder(value: string, length: number) {
  if (!value.trim()) return null;
  const parsed = value
    .split(/[,\\n]/)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => Number(chunk));
  if (parsed.length !== length) return null;
  return parsed;
}

type TinyGpsExerciseSetupProps = {
  state: TinyGpsState | null;
  onApply: (params: { m?: number[]; b?: number[] }, order?: number[]) => void;
  embedded?: boolean;
};

export function TinyGpsExerciseSetup({ state, onApply, embedded = false }: TinyGpsExerciseSetupProps) {
  const [mInput, setMInput] = useState("-1, 1");
  const [bInput, setBInput] = useState("0, 0");
  const [orderInput, setOrderInput] = useState("");

  const classCount = state?.params.b.length ?? 0;
  const sampleCount = state?.sample_count ?? 0;
  const canApply = state?.mode === "1d" && classCount > 0;

  const parsedM = useMemo(() => parseNumberList(mInput), [mInput]);
  const parsedB = useMemo(() => parseNumberList(bInput), [bInput]);
  const parsedOrder = useMemo(() => (sampleCount ? parseOrder(orderInput, sampleCount) : null), [orderInput, sampleCount]);

  const content = (
    <>
      {!embedded && <h3>Exercise Setup</h3>}
      <p className="panel-subtle">
        Paste the book’s initial parameters and sample order for the table walkthrough. (1D only.)
      </p>
      <div className="backprop-controls">
        <label>
          <span>m (comma-separated)</span>
          <input
            type="text"
            value={mInput}
            onChange={(event) => setMInput(event.target.value)}
            disabled={!canApply}
          />
        </label>
        <label>
          <span>b (comma-separated)</span>
          <input
            type="text"
            value={bInput}
            onChange={(event) => setBInput(event.target.value)}
            disabled={!canApply}
          />
        </label>
        <label>
          <span>Sample order (optional indices)</span>
          <input
            type="text"
            value={orderInput}
            onChange={(event) => setOrderInput(event.target.value)}
            placeholder="0,1,2,3"
            disabled={!canApply}
          />
        </label>
        <button
          type="button"
          className="ghost"
          onClick={() => onApply({ m: parsedM, b: parsedB }, parsedOrder ?? undefined)}
          disabled={!canApply || parsedM.length !== classCount || parsedB.length !== classCount}
        >
          Apply Exercise Params
        </button>
      </div>
    </>
  );

  if (embedded) return <div className="tinygps-setup">{content}</div>;

  return (
    <div className="panel backprop-card">
      {content}
    </div>
  );
}
