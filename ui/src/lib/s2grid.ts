import type { FeatureCollection, Feature, Polygon } from 'geojson';
import type { BBox } from './geo.ts';

// ---------------------------------------------------------------------------
// Compact UTM (Transverse Mercator) forward/inverse — WGS84 ellipsoid.
// Accurate to ~1 mm, sufficient for 10 m grid snapping.
// ---------------------------------------------------------------------------

const a  = 6378137.0;           // semi-major axis
const f  = 1 / 298.257223563;   // flattening
const b  = a * (1 - f);
const e2 = 1 - (b * b) / (a * a);
const e  = Math.sqrt(e2);
const k0 = 0.9996;              // UTM scale factor

function utmCentralMeridian(zone: number): number {
  return ((zone - 1) * 6 - 180 + 3) * Math.PI / 180;
}

export function utmZone(lng: number, lat: number): { zone: number; isSouth: boolean } {
  return { zone: Math.floor((lng + 180) / 6) + 1, isSouth: lat < 0 };
}

export function toUtm(lng: number, lat: number, zone: number, isSouth: boolean): { e: number; n: number } {
  const phi  = lat * Math.PI / 180;
  const lam  = lng * Math.PI / 180;
  const lam0 = utmCentralMeridian(zone);

  const N = a / Math.sqrt(1 - e2 * Math.sin(phi) ** 2);
  const T = Math.tan(phi) ** 2;
  const C = (e2 / (1 - e2)) * Math.cos(phi) ** 2;
  const A = Math.cos(phi) * (lam - lam0);

  const e4 = e2 * e2, e6 = e4 * e2;
  const M = a * (
    (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
    - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * Math.sin(2 * phi)
    + (15 * e4 / 256 + 45 * e6 / 1024) * Math.sin(4 * phi)
    - (35 * e6 / 3072) * Math.sin(6 * phi)
  );

  const easting = k0 * N * (
    A + (1 - T + C) * A ** 3 / 6
    + (5 - 18 * T + T * T + 72 * C - 58 * (e2 / (1 - e2))) * A ** 5 / 120
  ) + 500000;

  let northing = k0 * (M + N * Math.tan(phi) * (
    A ** 2 / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
    + (61 - 58 * T + T * T + 600 * C - 330 * (e2 / (1 - e2))) * A ** 6 / 720
  ));
  if (isSouth) northing += 10_000_000;

  return { e: easting, n: northing };
}

export function fromUtm(easting: number, northing: number, zone: number, isSouth: boolean): { lng: number; lat: number } {
  const x = easting - 500000;
  const y = isSouth ? northing - 10_000_000 : northing;
  const lam0 = utmCentralMeridian(zone);

  const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
  const M  = y / k0;
  const mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256));

  const phi1 = mu
    + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * Math.sin(2 * mu)
    + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * Math.sin(4 * mu)
    + (151 * e1 ** 3 / 96) * Math.sin(6 * mu)
    + (1097 * e1 ** 4 / 512) * Math.sin(8 * mu);

  const N1 = a / Math.sqrt(1 - e2 * Math.sin(phi1) ** 2);
  const T1 = Math.tan(phi1) ** 2;
  const C1 = (e2 / (1 - e2)) * Math.cos(phi1) ** 2;
  const R1 = a * (1 - e2) / (1 - e2 * Math.sin(phi1) ** 2) ** 1.5;
  const D  = x / (N1 * k0);

  const lat = phi1 - (N1 * Math.tan(phi1) / R1) * (
    D ** 2 / 2
    - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * (e2 / (1 - e2))) * D ** 4 / 24
    + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * (e2 / (1 - e2)) - 3 * C1 ** 2) * D ** 6 / 720
  );

  const lng = lam0 + (
    D
    - (1 + 2 * T1 + C1) * D ** 3 / 6
    + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * (e2 / (1 - e2)) + 24 * T1 ** 2) * D ** 5 / 120
  ) / Math.cos(phi1);

  return { lng: lng * 180 / Math.PI, lat: lat * 180 / Math.PI };
}

// ---------------------------------------------------------------------------
// Build S2 10 m pixel grid lines + snapped extent for a WGS84 bbox.
//
// Matches make_pixel_grid() in utils/pixel_collector.py:
//   - origin snapped with floor(x0 / r) * r  (pixel centres at snapped coords)
//   - pixel footprint extends ± HALF from each centre
//   - any pixel whose centre falls within [x0_snap, x1) × [y0_snap, y1) is collected
// ---------------------------------------------------------------------------
const CELL = 10;
const HALF = CELL / 2;

export interface S2GridData {
  gridLines: FeatureCollection;   // MultiLineString — the pixel grid
  pixelExtent: FeatureCollection; // Polygon — true footprint of collected pixels
}

export function buildS2Grid(bbox: BBox, properties: Record<string, unknown> = {}): S2GridData {
  const [minLng, minLat, maxLng, maxLat] = bbox;
  const midLng = (minLng + maxLng) / 2;
  const midLat = (minLat + maxLat) / 2;
  const { zone, isSouth } = utmZone(midLng, midLat);

  const corners = [
    toUtm(minLng, minLat, zone, isSouth),
    toUtm(maxLng, minLat, zone, isSouth),
    toUtm(maxLng, maxLat, zone, isSouth),
    toUtm(minLng, maxLat, zone, isSouth),
  ];
  const minE = Math.min(...corners.map(c => c.e));
  const maxE = Math.max(...corners.map(c => c.e));
  const minN = Math.min(...corners.map(c => c.n));
  const maxN = Math.max(...corners.map(c => c.n));

  // Match pixel_collector.py: floor snap for origin, collect centres up to (but not including) x1/y1
  const e0 = Math.floor(minE / CELL) * CELL;
  const n0 = Math.floor(minN / CELL) * CELL;
  // Last centre strictly less than maxE/maxN
  const eEnd = Math.ceil(maxE / CELL) * CELL - CELL;
  const nEnd = Math.ceil(maxN / CELL) * CELL - CELL;

  // Grid lines span from the western/southern edge of the first pixel to the
  // eastern/northern edge of the last pixel — i.e. full cell footprints.
  const lineMinE = e0 - HALF;
  const lineMaxE = eEnd + HALF;
  const lineMinN = n0 - HALF;
  const lineMaxN = nEnd + HALF;

  const lines: number[][][] = [];

  // Vertical lines at each pixel boundary (easting)
  for (let e = lineMinE; e <= lineMaxE + 0.01; e += CELL) {
    const p0 = fromUtm(e, lineMinN, zone, isSouth);
    const p1 = fromUtm(e, lineMaxN, zone, isSouth);
    lines.push([[p0.lng, p0.lat], [p1.lng, p1.lat]]);
  }

  // Horizontal lines at each pixel boundary (northing)
  for (let n = lineMinN; n <= lineMaxN + 0.01; n += CELL) {
    const p0 = fromUtm(lineMinE, n, zone, isSouth);
    const p1 = fromUtm(lineMaxE, n, zone, isSouth);
    lines.push([[p0.lng, p0.lat], [p1.lng, p1.lat]]);
  }

  // Snapped pixel extent polygon (corners of the full collected footprint)
  const sw = fromUtm(lineMinE, lineMinN, zone, isSouth);
  const se = fromUtm(lineMaxE, lineMinN, zone, isSouth);
  const ne = fromUtm(lineMaxE, lineMaxN, zone, isSouth);
  const nw = fromUtm(lineMinE, lineMaxN, zone, isSouth);
  const ring = [
    [sw.lng, sw.lat], [se.lng, se.lat], [ne.lng, ne.lat],
    [nw.lng, nw.lat], [sw.lng, sw.lat],
  ];

  return {
    gridLines: {
      type: 'FeatureCollection',
      features: lines.length === 0 ? [] : [{
        type: 'Feature',
        geometry: { type: 'MultiLineString', coordinates: lines },
        properties,
      }],
    },
    pixelExtent: {
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        geometry: { type: 'Polygon', coordinates: [ring] },
        properties,
      }],
    },
  };
}

export function emptyS2Grid(): S2GridData {
  const empty: FeatureCollection = { type: 'FeatureCollection', features: [] };
  return { gridLines: empty, pixelExtent: empty };
}

// ---------------------------------------------------------------------------
// Build chunk grid GeoJSON from sentinel2_tiles.geojson.
//
// Each S2 COG tile is 10980 × 10980 px at 10 m.  Chunks are 1024 × 1024 px
// (10240 m × 10240 m), starting from the tile's SW corner easting (x_left)
// and N edge northing (y_top) — both derivable by projecting the tile polygon
// corners to UTM and snapping to the nearest 10 m.
//
// The full grid is ceil(10980 / 1024) = 11 columns × 11 rows = up to 121
// chunks per tile, each rendered as a Polygon ring in WGS84.
// ---------------------------------------------------------------------------

const CHUNK_PX   = 1024;
const PIXEL_M    = 10;
const CHUNK_M    = CHUNK_PX * PIXEL_M;   // 10240 m
const TILE_PX    = 10980;
const CHUNK_COLS = Math.ceil(TILE_PX / CHUNK_PX);  // 11
const CHUNK_ROWS = Math.ceil(TILE_PX / CHUNK_PX);  // 11

export function buildChunkGrid(tilesGeoJSON: FeatureCollection): FeatureCollection {
  const features: Feature<Polygon>[] = [];

  for (const feat of tilesGeoJSON.features) {
    const geom = feat.geometry as any;
    const name = (feat.properties as any)?.name as string | undefined;
    if (!geom || !name) continue;

    // Collect all ring coordinates (handle Polygon and MultiPolygon).
    const rings: number[][][] =
      geom.type === 'MultiPolygon'
        ? (geom.coordinates as number[][][][]).flatMap((p: number[][][]) => p)
        : (geom.coordinates as number[][][]);
    const allCoords = rings.flat();
    if (allCoords.length === 0) continue;

    // Derive UTM zone from tile ID (first 1-2 digits).
    const zoneMatch = name.match(/^(\d{1,2})/);
    if (!zoneMatch) continue;
    const zone = parseInt(zoneMatch[1], 10);

    // Determine hemisphere from the MGRS band letter (3rd character of tile ID).
    // Bands C–M are south of equator; N–X are north.
    const band = name[2];
    const isSouth = band >= 'C' && band <= 'M';

    // Project all polygon vertices to UTM and find SW/NE extremes.
    let minE = Infinity, maxE = -Infinity, minN = Infinity, maxN = -Infinity;
    for (const [lng, lat] of allCoords) {
      const { e, n } = toUtm(lng, lat, zone, isSouth);
      if (e < minE) minE = e;
      if (e > maxE) maxE = e;
      if (n < minN) minN = n;
      if (n > maxN) maxN = n;
    }

    // Snap to nearest 10 m — matches the COG transform exactly.
    const xLeft = Math.round(minE / PIXEL_M) * PIXEL_M;
    const yTop  = Math.round(maxN / PIXEL_M) * PIXEL_M;

    // Emit one Polygon per chunk cell.
    for (let row = 0; row < CHUNK_ROWS; row++) {
      for (let col = 0; col < CHUNK_COLS; col++) {
        const e0 = xLeft + col * CHUNK_M;
        const e1 = e0 + CHUNK_M;
        const n1 = yTop  - row * CHUNK_M;
        const n0 = n1 - CHUNK_M;
        const sw = fromUtm(e0, n0, zone, isSouth);
        const se = fromUtm(e1, n0, zone, isSouth);
        const ne = fromUtm(e1, n1, zone, isSouth);
        const nw = fromUtm(e0, n1, zone, isSouth);
        features.push({
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [[
              [sw.lng, sw.lat], [se.lng, se.lat],
              [ne.lng, ne.lat], [nw.lng, nw.lat],
              [sw.lng, sw.lat],
            ]],
          },
          properties: { tile: name, chunk_row: row, chunk_col: col },
        });
      }
    }
  }

  return { type: 'FeatureCollection', features };
}
