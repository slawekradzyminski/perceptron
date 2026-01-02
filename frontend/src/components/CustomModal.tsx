import type { CustomConfig } from "../types";
import { valueColor } from "../utils/visuals";

const GRID_OPTIONS = [1, 2, 3, 4, 5];

type CustomModalProps = {
  isOpen: boolean;
  customConfig: CustomConfig;
  onClose: () => void;
  onRowsChange: (rows: number) => void;
  onColsChange: (cols: number) => void;
  onAddSample: () => void;
  onApply: () => void;
  onRemoveSample: (index: number) => void;
  onToggleCell: (sampleIndex: number, row: number, col: number) => void;
  onLabelChange: (index: number, value: number) => void;
};

export function CustomModal({
  isOpen,
  customConfig,
  onClose,
  onRowsChange,
  onColsChange,
  onAddSample,
  onApply,
  onRemoveSample,
  onToggleCell,
  onLabelChange,
}: CustomModalProps) {
  if (!isOpen) return null;

  return (
    <div className="modal is-open" role="dialog" aria-modal="true">
      <div className="modal-backdrop" onClick={onClose} />
      <div className="modal-card">
        <div className="modal-head">
          <div>
            <p className="eyebrow">Custom dataset</p>
            <h2>Design your own switchboard</h2>
            <p className="subhead">Pick a grid, define patterns, and label them.</p>
          </div>
          <button className="ghost" type="button" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="modal-body">
          <div className="custom-controls">
            <label>
              <span>Rows</span>
              <select
                value={customConfig.rows}
                className="select"
                onChange={(event) => onRowsChange(Number(event.target.value))}
              >
                {GRID_OPTIONS.map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Cols</span>
              <select
                value={customConfig.cols}
                className="select"
                onChange={(event) => onColsChange(Number(event.target.value))}
              >
                {GRID_OPTIONS.map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>
            <button type="button" onClick={onAddSample}>Add sample</button>
            <button type="button" className="primary" onClick={onApply}>
              Apply custom dataset
            </button>
          </div>
          <div className="custom-samples">
            {customConfig.samples.map((sample, index) => (
              <div className="sample-card" key={`sample-${index}`}>
                <div className="sample-head">
                  <span>Sample {index + 1}</span>
                  <div className="sample-actions">
                    <select
                      value={sample.y}
                      className="select small"
                      onChange={(event) => onLabelChange(index, Number(event.target.value))}
                    >
                      <option value={1}>+1</option>
                      <option value={-1}>-1</option>
                    </select>
                    <button
                      type="button"
                      onClick={() => onRemoveSample(index)}
                      disabled={customConfig.samples.length <= 1}
                    >
                      Remove
                    </button>
                  </div>
                </div>
                <div
                  className="sample-grid"
                  style={{ gridTemplateColumns: `repeat(${customConfig.cols}, 1fr)` }}
                >
                  {sample.grid.map((row, r) =>
                    row.map((cell, c) => (
                      <button
                        key={`cell-${index}-${r}-${c}`}
                        type="button"
                        className="sample-cell"
                        style={{ backgroundColor: valueColor(cell) }}
                        onClick={() => onToggleCell(index, r, c)}
                        title={cell.toString()}
                      />
                    )),
                  )}
                </div>
              </div>
            ))}
          </div>
          <p className="custom-hint">
            Click cells to toggle between -1 and +1. Labels define the expected output.
          </p>
        </div>
      </div>
    </div>
  );
}
