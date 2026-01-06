import { useMemo, useState } from "react";
import type { TinyGpsState } from "../../../types";

function parseNumberList(value: string) {
  if (!value.trim()) return [] as number[];
  return value
    .split(/[,\n]/)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => Number(chunk));
}

function parseOrder(value: string, length: number) {
  if (!value.trim()) return null;
  const parsed = value
    .split(/[,\n]/)
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk) => Number(chunk));
  if (parsed.length !== length) return null;
  return parsed;
}

function formatVec(values: number[]): string {
  return values.map((v) => v.toFixed(4)).join(", ");
}

type TinyGpsExerciseSetupProps = {
  state: TinyGpsState | null;
  onApply: (params: { m?: number[]; b?: number[] }, order?: number[]) => void;
  embedded?: boolean;
};

export function TinyGpsExerciseSetup({ state, onApply, embedded = false }: TinyGpsExerciseSetupProps) {
  const mode = state?.mode ?? "1d";
  const classCount = state?.params.b.length ?? 0;
  const sampleCount = state?.sample_count ?? 0;

  // Track which dataset we've initialized for, to reset inputs on dataset change
  const [lastDataset, setLastDataset] = useState<string | null>(null);
  const [mInput, setMInput] = useState("");
  const [bInput, setBInput] = useState("");
  const [orderInput, setOrderInput] = useState("");

  // When dataset changes, reset inputs to match new state params
  const currentDataset = state?.dataset ?? null;
  if (currentDataset !== lastDataset && state) {
    setLastDataset(currentDataset);
    if (mode === "1d" && state.params.m) {
      setMInput(formatVec(state.params.m));
    } else {
      setMInput("");
    }
    setBInput(formatVec(state.params.b));
    setOrderInput("");
  }

  const parsedM = useMemo(() => parseNumberList(mInput), [mInput]);
  const parsedB = useMemo(() => parseNumberList(bInput), [bInput]);
  const parsedOrder = useMemo(
    () => (sampleCount ? parseOrder(orderInput, sampleCount) : null),
    [orderInput, sampleCount],
  );

  // For 1D: need m and b; for 2D: only b (M matrix is handled separately)
  const canApply = classCount > 0;
  const mValid = mode === "2d" || parsedM.length === classCount;
  const bValid = parsedB.length === classCount;

  const content = (
    <>
      {!embedded && <h3>Exercise Setup</h3>}
      <p className="panel-subtle">
        Set initial parameters and sample order for the table walkthrough.
        {mode === "2d" && " (2D mode: only biases can be set here)"}
      </p>
      <div className="backprop-controls">
        {mode === "1d" && (
          <label>
            <span>m (comma-separated, {classCount} values)</span>
            <input
              type="text"
              value={mInput}
              onChange={(event) => setMInput(event.target.value)}
              disabled={!canApply}
              placeholder={`e.g. ${Array(classCount).fill("0").join(", ")}`}
            />
          </label>
        )}
        <label>
          <span>b (comma-separated, {classCount} values)</span>
          <input
            type="text"
            value={bInput}
            onChange={(event) => setBInput(event.target.value)}
            disabled={!canApply}
            placeholder={`e.g. ${Array(classCount).fill("0").join(", ")}`}
          />
        </label>
        <label>
          <span>Sample order (optional, {sampleCount} indices)</span>
          <input
            type="text"
            value={orderInput}
            onChange={(event) => setOrderInput(event.target.value)}
            placeholder={`e.g. ${Array.from({ length: Math.min(sampleCount, 4) }, (_, i) => i).join(",")}`}
            disabled={!canApply}
          />
        </label>
        <button
          type="button"
          className="ghost"
          onClick={() =>
            onApply(
              { m: mode === "1d" ? parsedM : undefined, b: parsedB },
              parsedOrder ?? undefined,
            )
          }
          disabled={!canApply || !mValid || !bValid}
        >
          Apply Exercise Params
        </button>
      </div>
    </>
  );

  if (embedded) return <div className="tinygps-setup">{content}</div>;

  return <div className="panel backprop-card">{content}</div>;
}
