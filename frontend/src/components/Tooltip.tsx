import type { TooltipState } from "../types";

type TooltipProps = {
  tooltip: TooltipState;
};

export function Tooltip({ tooltip }: TooltipProps) {
  if (!tooltip.visible) return null;
  return (
    <div className="cell-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
      {tooltip.text}
    </div>
  );
}
