import type { TinyGpsState } from "../../types";
import { formatVec } from "../common/formatters";
import { TinyGpsExerciseSetup } from "./TinyGpsExerciseSetup";

type TinyGpsStepSetupCardProps = {
  state: TinyGpsState | null;
  onApply: (params: { m?: number[]; b?: number[] }, order?: number[]) => void;
};

export function TinyGpsStepSetupCard({ state, onApply }: TinyGpsStepSetupCardProps) {
  return (
    <div className="panel backprop-card">
      <h3>TinyGPS Setup</h3>
      <div className="backprop-stack">
        <div>
          <span className="backprop-label">Current Params</span>
          <div className="backprop-mono">
            m={formatVec(state?.params.m ?? [], 4)} b={formatVec(state?.params.b ?? [], 4)}
          </div>
        </div>
      </div>
      <TinyGpsExerciseSetup state={state} onApply={onApply} embedded />
    </div>
  );
}
