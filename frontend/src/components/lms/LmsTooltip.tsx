export function LmsTooltip({
  tooltip,
}: {
  tooltip: { visible: boolean; text: string; x: number; y: number };
}) {
  return (
    <div
      className={`lms-tooltip ${tooltip.visible ? "visible" : ""}`}
      style={{ left: tooltip.x, top: tooltip.y }}
    >
      {tooltip.text}
    </div>
  );
}
