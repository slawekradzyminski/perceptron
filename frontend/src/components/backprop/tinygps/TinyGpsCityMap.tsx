import { useMemo, useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip, useMap, Polygon } from "react-leaflet";
import type { LatLngBoundsExpression, LatLngExpression } from "leaflet";
import type { TinyGpsState } from "../../../types";

type TinyGpsCityMapProps = {
  state: TinyGpsState | null;
};

const _fmt = (value: number, digits = 3) => value.toFixed(digits);

// City color scheme - matches book colors
const CITY_COLORS: Record<string, string> = {
  madrid: "#8b5cf6",    // Purple
  paris: "#e26b3c",     // Orange
  berlin: "#2d7dd2",    // Blue
  barcelona: "#10b981", // Green/Teal
  rome: "#f59e0b",
  london: "#ec4899",
  amsterdam: "#6366f1",
};

const getCityColor = (city: string) => CITY_COLORS[city.toLowerCase()] ?? "#6b7280";

// Component to fit bounds when points change
function FitBounds({ bounds }: { bounds: LatLngBoundsExpression }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(bounds, { padding: [30, 30] });
  }, [map, bounds]);
  return null;
}

// Compute logit for a given point
function computeLogit1D(lon: number, m: number, b: number): number {
  return m * lon + b;
}

// Paris center for normalization (matching backend)
const PARIS_CENTER_LAT = 48.8575;
const PARIS_CENTER_LON = 2.3514;

function computeLogit2D(lat: number, lon: number, M: number[], b: number): number {
  // Normalize coordinates: x1 = lon - paris_lon, x2 = lat - paris_lat
  const x1 = lon - PARIS_CENTER_LON;
  const x2 = lat - PARIS_CENTER_LAT;
  return M[0] * x1 + M[1] * x2 + b;
}

// Find which class wins at a given point
function winningClass1D(lon: number, m: number[], b: number[]): number {
  let maxLogit = -Infinity;
  let winner = 0;
  for (let k = 0; k < m.length; k++) {
    const logit = computeLogit1D(lon, m[k], b[k]);
    if (logit > maxLogit) {
      maxLogit = logit;
      winner = k;
    }
  }
  return winner;
}

function winningClass2D(lat: number, lon: number, M: number[][], b: number[]): number {
  let maxLogit = -Infinity;
  let winner = 0;
  for (let k = 0; k < M.length; k++) {
    const logit = computeLogit2D(lat, lon, M[k], b[k]);
    if (logit > maxLogit) {
      maxLogit = logit;
      winner = k;
    }
  }
  return winner;
}

type Boundary1D = {
  lon: number;
  leftClass: number;
  rightClass: number;
};

// Compute all decision boundaries for 1D (K-1 pairwise intersections that matter)
function computeBoundaries1D(m: number[], b: number[], lonRange: [number, number]): Boundary1D[] {
  const boundaries: Boundary1D[] = [];
  const K = m.length;
  
  // Find all pairwise intersections
  const intersections: { lon: number; i: number; j: number }[] = [];
  for (let i = 0; i < K; i++) {
    for (let j = i + 1; j < K; j++) {
      const denom = m[i] - m[j];
      if (Math.abs(denom) < 1e-9) continue;
      const lon = -(b[i] - b[j]) / denom;
      if (Number.isFinite(lon) && lon >= lonRange[0] && lon <= lonRange[1]) {
        intersections.push({ lon, i, j });
      }
    }
  }
  
  // Sort by longitude
  intersections.sort((a, b) => a.lon - b.lon);
  
  // Find actual decision boundaries (where argmax changes)
  for (const inter of intersections) {
    const leftOfBoundary = winningClass1D(inter.lon - 0.001, m, b);
    const rightOfBoundary = winningClass1D(inter.lon + 0.001, m, b);
    if (leftOfBoundary !== rightOfBoundary) {
      boundaries.push({
        lon: inter.lon,
        leftClass: leftOfBoundary,
        rightClass: rightOfBoundary,
      });
    }
  }
  
  // Deduplicate boundaries that are very close
  const uniqueBoundaries: Boundary1D[] = [];
  for (const b of boundaries) {
    if (!uniqueBoundaries.some(ub => Math.abs(ub.lon - b.lon) < 0.01)) {
      uniqueBoundaries.push(b);
    }
  }
  
  return uniqueBoundaries.sort((a, b) => a.lon - b.lon);
}

// For 2D: create a grid and find regions
type Region2D = {
  classIdx: number;
  city: string;
  polygon: LatLngExpression[];
  color: string;
};

function computeRegions2D(
  M: number[][],
  b: number[],
  cities: string[],
  latRange: [number, number],
  lonRange: [number, number],
  resolution: number = 30
): Region2D[] {
  const K = M.length;
  if (K === 0) return [];
  
  const latStep = (latRange[1] - latRange[0]) / resolution;
  const lonStep = (lonRange[1] - lonRange[0]) / resolution;
  
  // Create grid of winning classes
  const grid: number[][] = [];
  for (let i = 0; i <= resolution; i++) {
    const row: number[] = [];
    for (let j = 0; j <= resolution; j++) {
      const lat = latRange[0] + i * latStep;
      const lon = lonRange[0] + j * lonStep;
      row.push(winningClass2D(lat, lon, M, b));
    }
    grid.push(row);
  }
  
  // Simple approach: create polygons for each cell based on winning class
  // Group adjacent cells of same class into larger polygons would be better
  // but for now we'll create semi-transparent overlays per class
  
  const regions: Map<number, LatLngExpression[][]> = new Map();
  
  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      const classIdx = grid[i][j];
      const lat1 = latRange[0] + i * latStep;
      const lat2 = latRange[0] + (i + 1) * latStep;
      const lon1 = lonRange[0] + j * lonStep;
      const lon2 = lonRange[0] + (j + 1) * lonStep;
      
      const cell: LatLngExpression[] = [
        [lat1, lon1],
        [lat1, lon2],
        [lat2, lon2],
        [lat2, lon1],
      ];
      
      if (!regions.has(classIdx)) {
        regions.set(classIdx, []);
      }
      regions.get(classIdx)!.push(cell);
    }
  }
  
  // Convert to Region2D objects - merge cells into single polygon per class
  const result: Region2D[] = [];
  for (const [classIdx, cells] of regions) {
    const city = cities[classIdx] ?? `Class ${classIdx}`;
    // For simplicity, we'll render each cell as a separate polygon
    // A better approach would merge them, but this works for visualization
    for (const cell of cells) {
      result.push({
        classIdx,
        city,
        polygon: cell,
        color: getCityColor(city),
      });
    }
  }
  
  return result;
}

// Compute 1D classification regions (vertical stripes)
type Region1D = {
  classIdx: number;
  city: string;
  lonStart: number;
  lonEnd: number;
  color: string;
};

function computeRegions1D(
  m: number[],
  b: number[],
  cities: string[],
  lonRange: [number, number]
): Region1D[] {
  const boundaries = computeBoundaries1D(m, b, lonRange);
  const regions: Region1D[] = [];
  
  // Create regions between boundaries
  const breakpoints = [lonRange[0], ...boundaries.map(b => b.lon), lonRange[1]];
  
  for (let i = 0; i < breakpoints.length - 1; i++) {
    const midLon = (breakpoints[i] + breakpoints[i + 1]) / 2;
    const classIdx = winningClass1D(midLon, m, b);
    const city = cities[classIdx] ?? `Class ${classIdx}`;
    regions.push({
      classIdx,
      city,
      lonStart: breakpoints[i],
      lonEnd: breakpoints[i + 1],
      color: getCityColor(city),
    });
  }
  
  return regions;
}

export function TinyGpsCityMap({ state }: TinyGpsCityMapProps) {
  const samples = state?.samples;
  const params = state?.params;
  const mode = state?.mode;
  const stateCities = state?.cities;
  
  const cities = useMemo(() => stateCities ?? [], [stateCities]);

  const points = useMemo(() => {
    if (!samples) return [];
    return samples.map((sample) => ({
      city: sample.city,
      lat: sample.lat,
      lon: sample.lon,
    }));
  }, [samples]);

  // Calculate bounds for the map
  const mapBounds = useMemo((): { bounds: LatLngBoundsExpression; latRange: [number, number]; lonRange: [number, number] } | null => {
    if (points.length === 0) return null;
    const lats = points.map((p) => p.lat);
    const lons = points.map((p) => p.lon);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);
    const latPad = Math.max((maxLat - minLat) * 0.3, 2);
    const lonPad = Math.max((maxLon - minLon) * 0.3, 2);
    return {
      bounds: [
        [minLat - latPad, minLon - lonPad],
        [maxLat + latPad, maxLon + lonPad],
      ],
      latRange: [minLat - latPad, maxLat + latPad],
      lonRange: [minLon - lonPad, maxLon + lonPad],
    };
  }, [points]);

  // 1D boundaries
  const boundaries1D = useMemo(() => {
    if (mode !== "1d" || !params?.m || !params?.b || !mapBounds) return [];
    return computeBoundaries1D(params.m, params.b, mapBounds.lonRange);
  }, [mode, params, mapBounds]);

  // 1D regions
  const regions1D = useMemo(() => {
    if (mode !== "1d" || !params?.m || !params?.b || !mapBounds || cities.length === 0) return [];
    return computeRegions1D(params.m, params.b, cities, mapBounds.lonRange);
  }, [mode, params, mapBounds, cities]);

  // 2D regions
  const regions2D = useMemo(() => {
    if (mode !== "2d" || !params?.M || !params?.b || !mapBounds || cities.length === 0) return [];
    return computeRegions2D(params.M, params.b, cities, mapBounds.latRange, mapBounds.lonRange, 25);
  }, [mode, params, mapBounds, cities]);

  if (!state) {
    return (
      <div className="panel backprop-card tinygps-map-card">
        <h3>City Map + Boundary</h3>
        <p className="diag-placeholder">Load TinyGPS state to view the map.</p>
      </div>
    );
  }

  const stateSamples = state.samples ?? [];
  const uniqueCities = [...new Set(stateSamples.map((sample) => sample.city))];

  if (points.length === 0 || !mapBounds) {
    return (
      <div className="panel backprop-card tinygps-map-card">
        <h3>City Map + Boundary</h3>
        <p className="diag-placeholder">No data points available.</p>
      </div>
    );
  }

  // Default center (middle of Europe)
  const defaultCenter: [number, number] = [50, 8];

  return (
    <div className="panel backprop-card tinygps-map-card">
      <h3>City Map + Boundaries ({mode === "1d" ? "1D" : "2D"} · {cities.length} classes)</h3>

      {/* Map Container */}
      <div className="tinygps-leaflet-container">
        <MapContainer
          center={defaultCenter}
          zoom={5}
          scrollWheelZoom={true}
          className="tinygps-leaflet-map"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <FitBounds bounds={mapBounds.bounds} />

          {/* 1D Classification regions (vertical stripes) */}
          {mode === "1d" && regions1D.map((region, idx) => (
            <Polygon
              key={`region-1d-${idx}`}
              positions={[
                [mapBounds.latRange[0] - 10, region.lonStart],
                [mapBounds.latRange[0] - 10, region.lonEnd],
                [mapBounds.latRange[1] + 10, region.lonEnd],
                [mapBounds.latRange[1] + 10, region.lonStart],
              ]}
              pathOptions={{
                color: region.color,
                weight: 0,
                fillColor: region.color,
                fillOpacity: 0.15,
              }}
            />
          ))}

          {/* 2D Classification regions (grid cells) */}
          {mode === "2d" && regions2D.map((region, idx) => (
            <Polygon
              key={`region-2d-${idx}`}
              positions={region.polygon}
              pathOptions={{
                color: region.color,
                weight: 0,
                fillColor: region.color,
                fillOpacity: 0.2,
              }}
            />
          ))}

          {/* City markers */}
          {points.map((point, idx) => {
            const color = getCityColor(point.city);
            return (
              <CircleMarker
                key={`${point.city}-${idx}`}
                center={[point.lat, point.lon]}
                radius={10}
                pathOptions={{
                  fillColor: color,
                  color: "#fff",
                  weight: 3,
                  opacity: 1,
                  fillOpacity: 0.9,
                }}
              >
                <Tooltip direction="top" offset={[0, -8]}>
                  <div style={{ textAlign: "center" }}>
                    <strong style={{ color, textTransform: "uppercase" }}>
                      {point.city}
                    </strong>
                    <br />
                    <span style={{ fontFamily: "monospace", fontSize: 11 }}>
                      {point.lat.toFixed(4)}°N, {point.lon.toFixed(4)}°E
                    </span>
                  </div>
                </Tooltip>
              </CircleMarker>
            );
          })}

          {/* 1D Decision boundary lines */}
          {mode === "1d" && boundaries1D.map((boundary, idx) => (
            <Polyline
              key={`boundary-${idx}`}
              positions={[
                [mapBounds.latRange[0] - 10, boundary.lon],
                [mapBounds.latRange[1] + 10, boundary.lon],
              ]}
              pathOptions={{
                color: "#1c1b19",
                weight: 3,
                dashArray: "8, 6",
                opacity: 0.8,
              }}
            >
              <Tooltip direction="top" sticky>
                <div style={{ textAlign: "center" }}>
                  <strong style={{ color: "#1c1b19" }}>Decision Boundary</strong>
                  <br />
                  <span style={{ fontFamily: "monospace", fontSize: 11 }}>
                    {cities[boundary.leftClass]} ↔ {cities[boundary.rightClass]}
                  </span>
                  <br />
                  <span style={{ fontFamily: "monospace", fontSize: 11 }}>
                    Lon: {boundary.lon.toFixed(4)}°E
                  </span>
                </div>
              </Tooltip>
            </Polyline>
          ))}
        </MapContainer>
      </div>

      {/* Legend */}
      <div className="tinygps-map-legend">
        {uniqueCities.map((city) => (
          <div key={city} className="tinygps-legend-item">
            <span
              className="tinygps-legend-dot"
              style={{ backgroundColor: getCityColor(city) }}
            />
            <span className="tinygps-legend-label">{city}</span>
          </div>
        ))}
        {boundaries1D.length > 0 && (
          <div className="tinygps-legend-item tinygps-legend-boundary">
            <span className="tinygps-legend-line" />
            <span className="tinygps-legend-label">
              {boundaries1D.length} {boundaries1D.length === 1 ? "boundary" : "boundaries"}
            </span>
          </div>
        )}
      </div>

      <div className="diag-note">
        {points.length} data points across {uniqueCities.length} cities
        {mode === "1d" && boundaries1D.length > 0 && (
          <> · {boundaries1D.length} decision {boundaries1D.length === 1 ? "boundary" : "boundaries"}</>
        )}
        {mode === "2d" && <> · 2D plane classification</>}
      </div>
    </div>
  );
}
