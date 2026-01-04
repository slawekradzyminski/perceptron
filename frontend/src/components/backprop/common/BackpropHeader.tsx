type BackpropHeaderProps = {
  loading: boolean;
  hasState: boolean;
  onStepPrimary: () => void;
  onStepSecondary?: () => void;
  primaryLabel: string;
  secondaryLabel?: string;
};

export function BackpropHeader({
  loading,
  hasState,
  onStepPrimary,
  onStepSecondary,
  primaryLabel,
  secondaryLabel,
}: BackpropHeaderProps) {
  return (
    <div className="backprop-head">
      <div>
        <h2>Backprop Lab (Chapter 3)</h2>
        <p className="panel-subtle">Step through TinyGPS or regression exercises with explicit gradients.</p>
      </div>
      <div className="backprop-actions">
        <button type="button" onClick={onStepPrimary} disabled={loading || !hasState}>
          {primaryLabel}
        </button>
        {secondaryLabel && onStepSecondary && (
          <button type="button" onClick={onStepSecondary} disabled={loading || !hasState}>
            {secondaryLabel}
          </button>
        )}
      </div>
    </div>
  );
}
