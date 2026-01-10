import type { FilterGrid } from "../../hooks/alexnet/useAlexNetApi";

type AlexNetFilterSectionProps = {
  filters: FilterGrid | null;
  selectedLayer: number;
  hoveredFilter: number | null;
  onHoverChange: (value: number | null) => void;
};

export function AlexNetFilterSection({
  filters,
  selectedLayer,
  hoveredFilter,
  onHoverChange,
}: AlexNetFilterSectionProps) {
  return (
    <div className="viz-section">
      <h3>
        Pattern Detectors (Layer {selectedLayer} Filters)
        {selectedLayer === 1 && (
          <span className="hint"> — Edge and color blob detectors</span>
        )}
        {selectedLayer === 2 && (
          <span className="hint"> — Corner and texture detectors</span>
        )}
        {selectedLayer >= 3 && (
          <span className="hint"> — High-level feature detectors</span>
        )}
      </h3>
      <div className="filter-grid">
        {filters?.filters.map((filter) => (
          <div
            key={filter.index}
            className={`filter-cell ${
              hoveredFilter === filter.index ? "hovered" : ""
            }`}
            onMouseEnter={() => onHoverChange(filter.index)}
            onMouseLeave={() => onHoverChange(null)}
            title={`Filter ${filter.index}`}
          >
            <img
              src={`data:image/png;base64,${filter.image}`}
              alt={`Filter ${filter.index}`}
            />
            <span className="filter-index">{filter.index}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
