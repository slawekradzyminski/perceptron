import type { TinyGpsState } from "../../types";

type TinyGpsCityMapProps = {
  state: TinyGpsState | null;
};

const fmt = (value: number, digits = 3) => value.toFixed(digits);

export function TinyGpsCityMap({ state }: TinyGpsCityMapProps) {
  if (!state) {
    return (
      <div className="panel backprop-card">
        <h3>City Map</h3>
        <p className="diag-placeholder">Load TinyGPS state to view the map.</p>
      </div>
    );
  }

  const samples = state.samples ?? [];
  const cities = new Set(samples.map((sample) => sample.city));
  if (!cities.has("paris") || !cities.has("berlin") || cities.size !== 2 || state.mode !== "1d") {
    return (
      <div className="panel backprop-card">
        <h3>City Map</h3>
        <p className="diag-placeholder">City map is available for 2-city 1D datasets (Paris/Berlin).</p>
      </div>
    );
  }

  const points = samples.map((sample) => ({
    city: sample.city,
    lat: sample.lat,
    lon: sample.lon,
  }));

  const lats = points.map((p) => p.lat);
  const lons = points.map((p) => p.lon);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const pad = 0.2;
  const latPad = (maxLat - minLat) * pad || 0.1;
  const lonPad = (maxLon - minLon) * pad || 0.1;

  const width = 520;
  const height = 220;
  const innerPad = 30;
  const lonMin = minLon - lonPad;
  const lonMax = maxLon + lonPad;
  const latMin = minLat - latPad;
  const latMax = maxLat + latPad;

  const projectX = (lon: number) =>
    innerPad + ((lon - lonMin) / (lonMax - lonMin || 1)) * (width - innerPad * 2);
  const projectY = (lat: number) =>
    innerPad + (1 - (lat - latMin) / (latMax - latMin || 1)) * (height - innerPad * 2);

  const [m1, m2] = state.params.m ?? [];
  const [b1, b2] = state.params.b ?? [];
  const denom = m1 - m2;
  const boundaryLon = Math.abs(denom) < 1e-6 ? null : -(b1 - b2) / denom;
  const boundaryX = boundaryLon === null ? null : projectX(boundaryLon);
  const boundaryInside =
    boundaryLon !== null && boundaryLon >= lonMin && boundaryLon <= lonMax && Number.isFinite(boundaryLon);

  return (
    <div className="panel backprop-card tinygps-map-card">
      <h3>City Map + Boundary</h3>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Paris Berlin map">
        <rect x={0} y={0} width={width} height={height} rx={14} fill="#fffdf8" />
        <rect
          x={innerPad}
          y={innerPad}
          width={width - innerPad * 2}
          height={height - innerPad * 2}
          fill="#fff9f2"
          stroke="#e9dccb"
        />
        {boundaryX !== null && (
          <line
            x1={Math.min(width - innerPad, Math.max(innerPad, boundaryX))}
            y1={innerPad}
            x2={Math.min(width - innerPad, Math.max(innerPad, boundaryX))}
            y2={height - innerPad}
            stroke={boundaryInside ? "#e26b3c" : "#cbb8a4"}
            strokeWidth={2}
            strokeDasharray="6 6"
          />
        )}
        {points.map((point, idx) => {
          const x = projectX(point.lon);
          const y = projectY(point.lat);
          const fill = point.city === "paris" ? "#e26b3c" : "#2d7dd2";
          return (
            <g key={`${point.city}-${idx}`}>
              <title>
                {point.city} ({fmt(point.lat, 4)}, {fmt(point.lon, 4)})
              </title>
              <circle cx={x} cy={y} r={4.5} fill={fill} />
            </g>
          );
        })}
        <text x={innerPad + 4} y={innerPad - 8} fontSize={10} fill="#7b6a57">
          lat ↑
        </text>
        <text x={width - innerPad - 30} y={height - 20} fontSize={10} fill="#7b6a57">
          lon →
        </text>
        <text x={innerPad} y={innerPad + 10} fontSize={9} fill="#9c8975">
          {fmt(latMax, 3)}
        </text>
        <text x={innerPad} y={height - innerPad + 12} fontSize={9} fill="#9c8975">
          {fmt(latMin, 3)}
        </text>
        <text x={innerPad} y={height - 6} fontSize={9} fill="#9c8975">
          {fmt(lonMin, 3)}
        </text>
        <text x={width - innerPad} y={height - 6} fontSize={9} fill="#9c8975" textAnchor="end">
          {fmt(lonMax, 3)}
        </text>
      </svg>
      <div className="diag-note">
        {boundaryLon === null
          ? "Boundary undefined (m1 ≈ m2)."
          : `Boundary at lon ≈ ${fmt(boundaryLon, 4)}${boundaryInside ? "" : " (outside view)"}`}
      </div>
    </div>
  );
}
