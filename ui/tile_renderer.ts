/**
 * tile_renderer.ts — On-the-fly ranking tile renderer.
 *
 * Binary format v2 (.bin): 64-byte header + N×8-byte records.
 * Detected by magic number BLN2 (0x424C4E32) at byte 0.
 * See HEADER_BYTES_V2 block below for full layout.
 *
 * Legacy v1 (40-byte header, no magic): used by S1 tiles from s1_tile_builder.py.
 *
 * The .bin is built transparently from the CSV on first request and cached on
 * disk. In memory only the typed arrays (~17 MB for 2M pixels) are held.
 */

import { join, dirname, fromFileUrl } from "jsr:@std/path";

const __dirname = dirname(fromFileUrl(import.meta.url));
const OUTPUTS_DIR = join(__dirname, "..", "outputs");
const SCORES_DIR  = join(OUTPUTS_DIR, "scores");

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

interface Grid {
  keys: Uint32Array;  // sorted cell indices (yi*width + xi)
  vals: Float32Array; // corresponding prob values
  // UTM-based lookup (exact)
  utmOriginE: number; // easting of xi=0 pixel centre
  utmOriginN: number; // northing of yi=0 pixel centre (standard UTM, includes false northing)
  res: number;        // pixel spacing in metres (10.0)
  utmZone: number;    // positive = north, negative = south
  width: number;
  height: number;
  // WGS84 fallback (kept for S1 grids built without UTM metadata)
  lonMin: number;
  latMax: number;
  resX: number;
  resY: number;
}

const GRID_CACHE_MAX = 3;
const gridCache = new Map<string, Grid>();
const gridLoading = new Map<string, Promise<Grid | null>>(); // in-flight loads

function cacheSet(key: string, grid: Grid): void {
  gridCache.delete(key);
  gridCache.set(key, grid);
  if (gridCache.size > GRID_CACHE_MAX) {
    gridCache.delete(gridCache.keys().next().value!);
  }
}

// ---------------------------------------------------------------------------
// UTM forward projection (WGS84 → UTM)
// ---------------------------------------------------------------------------

// Returns [easting, northing] in metres (standard UTM with false northing).
// Southern hemisphere: northing includes the 10,000,000 m false northing
// so values are large positive numbers (~8,000,000–10,000,000 for Australia).
function wgs84ToUtm(latDeg: number, lonDeg: number, zone: number, south: boolean): [number, number] {
  const a  = 6378137.0;
  const f  = 1 / 298.257223563;
  const b  = a * (1 - f);
  const e2 = 1 - (b / a) ** 2;
  const ep2 = e2 / (1 - e2);
  const k0 = 0.9996;

  const lat = latDeg  * Math.PI / 180;
  const lon0 = ((zone - 1) * 6 - 180 + 3) * Math.PI / 180;
  const lon = lonDeg * Math.PI / 180;

  const N  = a / Math.sqrt(1 - e2 * Math.sin(lat) ** 2);
  const T  = Math.tan(lat) ** 2;
  const C  = ep2 * Math.cos(lat) ** 2;
  const A  = Math.cos(lat) * (lon - lon0);
  const M  = a * (
    (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
    - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * Math.sin(2 * lat)
    + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * Math.sin(4 * lat)
    - (35 * e2 ** 3 / 3072) * Math.sin(6 * lat)
  );

  const E = k0 * N * (
    A + (1 - T + C) * A ** 3 / 6
    + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120
  ) + 500000;

  let Nval = k0 * (
    M + N * Math.tan(lat) * (
      A ** 2 / 2
      + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
      + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720
    )
  );
  if (south) Nval += 10_000_000; // false northing for southern hemisphere

  return [E, Nval];
}

// ---------------------------------------------------------------------------
// .bin build + load
// ---------------------------------------------------------------------------

// New header layout (64 bytes), detected by magic number:
//   [0]  u32 magic = 0x424C4E32 ("BLN2")
//   [4]  i32 utmZone  (negative = southern hemisphere; 0 = WGS84 fallback only)
//   [8]  f64 utmOriginE
//   [16] f64 utmOriginN
//   [24] f64 res          (pixel spacing in metres)
//   [32] u32 width
//   [36] u32 height
//   [40] f64 lonMin   (WGS84 fallback / info)
//   [48] f64 latMin   (WGS84 fallback / info)
//   [56] f64 latMax   (WGS84 fallback / info)
// Total: 64 bytes
//
// Old format (40 bytes, no magic): f64×4 + u32×2 = lonMin,latMax,resX,resY,width,height
// Detected by: first 4 bytes reinterpreted as u32 ≠ BLN2_MAGIC.
const HEADER_BYTES_V2 = 64;
const HEADER_BYTES_V1 = 40;
const BLN2_MAGIC = 0x424C4E32;
// Export for use in loadBin size check
const HEADER_BYTES = HEADER_BYTES_V2;

function binPath(location: string, stem: string): string {
  return join(SCORES_DIR, location, `${stem}.bin`);
}

async function buildBin(csvPath: string, outPath: string): Promise<void> {
  console.log(`Building binary cache: ${outPath}`);
  const t0 = performance.now();

  async function streamLines(cb: (cols: string[], lineNo: number) => void): Promise<void> {
    const file = await Deno.open(csvPath, { read: true });
    const decoder = new TextDecoder();
    let buf = "";
    let lineNo = 0;
    for await (const chunk of file.readable) {
      buf += decoder.decode(chunk, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, nl).trimEnd();
        buf = buf.slice(nl + 1);
        cb(line.split(","), lineNo++);
      }
    }
    if (buf.trimEnd()) cb(buf.trimEnd().split(","), lineNo);
  }

  // Pass 0: detect columns
  let iId = -1, iLon = -1, iLat = -1, iProb = -1;
  await streamLines((cols, lineNo) => {
    if (lineNo !== 0) return;
    iId   = cols.indexOf("point_id");
    iLon  = cols.indexOf("lon");
    iLat  = cols.indexOf("lat");
    iProb = cols.findIndex((c) => c.startsWith("prob_"));
  });
  if (iLon < 0 || iLat < 0 || iProb < 0) throw new Error(`Missing columns in ${csvPath}`);

  // Pass 1: grid dimensions and corner pixel coordinates from point_id.
  // We need the actual lon/lat of the NW corner pixel (xi=0, yi=yiMax) to use
  // as the grid origin, and the NE/SW corners to derive resX/resY independently.
  // Using overall min/max lon/lat is wrong because the grid is not axis-aligned
  // in WGS84 (UTM→WGS84 reprojection means rows/cols are slightly diagonal).
  let xiMax = -1, yiMax = -1;
  let lonMin = Infinity, lonMax = -Infinity, latMin = Infinity, latMax = -Infinity;
  // Corner pixel coords: NW=(xi=0,yi=yiMax), NE=(xi=xiMax,yi=yiMax), SW=(xi=0,yi=0)
  let lonNW = NaN, latNW = NaN, lonNE = NaN, latNE = NaN, lonSW = NaN, latSW = NaN;
  await streamLines((cols, lineNo) => {
    if (lineNo === 0) return;
    const lon = parseFloat(cols[iLon]);
    const lat = parseFloat(cols[iLat]);
    if (isNaN(lon) || isNaN(lat)) return;
    if (lon < lonMin) lonMin = lon;
    if (lon > lonMax) lonMax = lon;
    if (lat < latMin) latMin = lat;
    if (lat > latMax) latMax = lat;
    if (iId >= 0) {
      const parts = cols[iId].split("_");
      if (parts.length >= 3) {
        const xi = parseInt(parts[1]), yi = parseInt(parts[2]);
        if (xi > xiMax) xiMax = xi;
        if (yi > yiMax) yiMax = yi;
      }
    }
  });
  // Second pass to get corner coords (need yiMax known first)
  if (iId >= 0 && xiMax > 0 && yiMax > 0) {
    await streamLines((cols, lineNo) => {
      if (lineNo === 0) return;
      const parts = cols[iId].split("_");
      if (parts.length < 3) return;
      const xi = parseInt(parts[1]), yi = parseInt(parts[2]);
      const lon = parseFloat(cols[iLon]), lat = parseFloat(cols[iLat]);
      if (xi === 0    && yi === yiMax) { lonNW = lon; latNW = lat; }
      if (xi === xiMax && yi === yiMax) { lonNE = lon; latNE = lat; }
      if (xi === 0    && yi === 0    ) { lonSW = lon; latSW = lat; }
    });
  }

  // Use point_id grid dimensions when available; fall back to coordinate-derived grid.
  let width: number, height: number, resX: number, resY: number;
  let utmOriginE = 0, utmOriginN = 0, utmZone = 0, res = 10.0;
  if (iId >= 0 && xiMax > 0 && yiMax > 0 && !isNaN(lonNW) && !isNaN(lonNE) && !isNaN(latSW)) {
    width  = xiMax + 1;
    height = yiMax + 1;
    // resX/resY kept for WGS84 fallback path only
    resX = (lonNE - lonNW) / xiMax;
    resY = (latNW - latSW) / yiMax;
    // Use NW corner as origin for WGS84 fallback
    lonMin = lonNW;
    latMax = latNW;
    // Compute UTM origin from SW corner pixel (xi=0, yi=0)
    // and infer zone from grid centre lon.
    const centreLon = (lonNW + lonNE) / 2;
    utmZone = Math.floor((centreLon + 180) / 6) + 1;
    const south = latSW < 0;
    if (south) utmZone = -utmZone; // negative = southern hemisphere
    const [swE, swN] = wgs84ToUtm(latSW, lonSW, Math.abs(utmZone), south);
    utmOriginE = swE;
    utmOriginN = swN;
    // Infer res from NW-SW northing span
    const [nwE, nwN] = wgs84ToUtm(latNW, lonNW, Math.abs(utmZone), south);
    res = Math.round((nwN - swN) / yiMax);  // should be ~10
    if (res <= 0) res = 10;
  } else {
    resX = resY = 0.0001;
    width  = Math.round((lonMax - lonMin) / resX) + 1;
    height = Math.round((latMax - latMin) / resY) + 1;
  }

  // Count valid rows for pre-allocation
  let count = 0;
  await streamLines((cols, lineNo) => {
    if (lineNo === 0) return;
    if (!isNaN(parseFloat(cols[iProb]))) count++;
  });

  // Pass 2: fill pre-allocated typed arrays
  // yi is stored as the raw pixel_collector yi (0 = southernmost, yiMax = northernmost).
  const keys = new Uint32Array(count);
  const vals = new Float32Array(count);
  let idx = 0;
  await streamLines((cols, lineNo) => {
    if (lineNo === 0) return;
    const prob = parseFloat(cols[iProb]);
    if (isNaN(prob)) return;
    let xi: number, yi: number;
    if (iId >= 0 && xiMax > 0) {
      const parts = cols[iId].split("_");
      xi = parseInt(parts[1]);
      yi = parseInt(parts[2]); // raw: 0 = southernmost
    } else {
      const lon = parseFloat(cols[iLon]);
      const lat = parseFloat(cols[iLat]);
      xi = Math.round((lon - lonMin) / resX);
      yi = Math.round((lat - latMin) / resY); // latMin = SW lat
    }
    keys[idx] = yi * width + xi;
    vals[idx] = prob;
    idx++;
  });

  // Sort by key using an index array, then reorder in-place
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;
  order.sort((a, b) => keys[a] - keys[b]);

  const sortedKeys = new Uint32Array(count);
  const sortedVals = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    sortedKeys[i] = keys[order[i]];
    sortedVals[i] = vals[order[i]];
  }

  // Write binary file (64-byte v2 header)
  const buf = new ArrayBuffer(HEADER_BYTES_V2 + count * 8);
  const dv  = new DataView(buf);
  dv.setUint32(0,   BLN2_MAGIC,  true);
  dv.setInt32(4,    utmZone,     true);
  dv.setFloat64(8,  utmOriginE,  true);
  dv.setFloat64(16, utmOriginN,  true);
  dv.setFloat64(24, res,         true);
  dv.setUint32(32,  width,       true);
  dv.setUint32(36,  height,      true);
  dv.setFloat64(40, lonMin,      true);
  dv.setFloat64(48, latMin,      true);
  dv.setFloat64(56, latMax,      true);
  let off = HEADER_BYTES;
  for (let i = 0; i < count; i++) {
    dv.setUint32(off,      sortedKeys[i], true);
    dv.setFloat32(off + 4, sortedVals[i], true);
    off += 8;
  }
  const tmpPath = outPath + ".tmp";
  await Deno.writeFile(tmpPath, new Uint8Array(buf));
  await Deno.rename(tmpPath, outPath);
  console.log(`Binary cache built in ${(performance.now() - t0).toFixed(0)} ms  (${count} pixels, ${width}×${height})`);
}

async function loadBin(path: string): Promise<Grid> {
  const raw = await Deno.readFile(path);
  const dv  = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);

  const magic = dv.getUint32(0, true);
  const isV2  = magic === BLN2_MAGIC;
  const hdrBytes = isV2 ? HEADER_BYTES_V2 : HEADER_BYTES_V1;

  let utmOriginE: number, utmOriginN: number, res: number, utmZone: number;
  let width: number, height: number, lonMin: number, latMax: number, resX: number, resY: number;

  if (isV2) {
    utmZone    = dv.getInt32(4,    true);
    utmOriginE = dv.getFloat64(8,  true);
    utmOriginN = dv.getFloat64(16, true);
    res        = dv.getFloat64(24, true);
    width      = dv.getUint32(32,  true);
    height     = dv.getUint32(36,  true);
    lonMin     = dv.getFloat64(40, true);
    // latMin at [48] not needed at runtime
    latMax     = dv.getFloat64(56, true);
    resX = 0; resY = 0;
  } else {
    // Legacy v1 (S1 tiles, old score bins)
    utmOriginE = 0; utmOriginN = 0; res = 10; utmZone = 0;
    lonMin = dv.getFloat64(0,  true);
    latMax = dv.getFloat64(8,  true);
    resX   = dv.getFloat64(16, true);
    resY   = dv.getFloat64(24, true);
    width  = dv.getUint32(32,  true);
    height = dv.getUint32(36,  true);
  }

  const n = (raw.byteLength - hdrBytes) / 8;
  const keys = new Uint32Array(n);
  const vals = new Float32Array(n);
  let off = hdrBytes;
  for (let i = 0; i < n; i++) {
    keys[i] = dv.getUint32(off,     true);
    vals[i] = dv.getFloat32(off + 4, true);
    off += 8;
  }
  return { keys, vals, utmOriginE, utmOriginN, res, utmZone, lonMin, latMax, resX, resY, width, height };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function listRankings(): Record<string, Array<{ stem: string; label: string }>> {
  const results: Record<string, Array<{ stem: string; label: string }>> = {};
  try {
    for (const entry of Deno.readDirSync(SCORES_DIR)) {
      if (!entry.isDirectory) continue;
      const dir = join(SCORES_DIR, entry.name);
      const runs: Array<{ stem: string; label: string }> = [];
      try {
        for (const f of Deno.readDirSync(dir)) {
          if (f.isFile && f.name.endsWith(".csv")) {
            const stem = f.name.replace(/\.csv$/, "");
            runs.push({ stem, label: stem });
          }
        }
      } catch { /* unreadable subdir */ }
      if (runs.length > 0) {
        runs.sort((a, b) => a.stem.localeCompare(b.stem));
        results[entry.name] = runs;
      }
    }
  } catch { /* outputs dir may not exist */ }
  return results;
}

export function loadGrid(location: string, stem: string): Promise<Grid | null> {
  const cacheKey = `${location}/${stem}`;
  if (gridCache.has(cacheKey)) {
    const hit = gridCache.get(cacheKey)!;
    gridCache.delete(cacheKey);
    gridCache.set(cacheKey, hit);
    return Promise.resolve(hit);
  }

  // Return existing in-flight promise if already loading
  if (gridLoading.has(cacheKey)) return gridLoading.get(cacheKey)!;

  const promise = (async () => {
    const bin = binPath(location, stem);
    const csv = join(SCORES_DIR, location, `${stem}.csv`);

    let binExists = false;
    try {
      const st = Deno.statSync(bin);
      // A valid .bin has at least the header plus one record; a header-only file
      // means a previous buildBin was interrupted before the atomic rename.
      binExists = st.size > HEADER_BYTES_V2;
      if (!binExists) Deno.removeSync(bin);
    } catch { /* absent */ }
    if (!binExists) {
      try { Deno.statSync(csv); } catch { return null; }
      await buildBin(csv, bin);
    }

    const grid = await loadBin(bin);
    cacheSet(cacheKey, grid);
    console.log(`Grid loaded: ${location}/${stem} (${grid.keys.length} pixels)`);
    return grid;
  })().finally(() => gridLoading.delete(cacheKey));

  gridLoading.set(cacheKey, promise);
  return promise;
}

// ---------------------------------------------------------------------------
// Tile bounds
// ---------------------------------------------------------------------------

function tileBoundsWGS84(z: number, x: number, y: number) {
  const n = 2 ** z;
  const lonMin = (x / n) * 360 - 180;
  const lonMax = ((x + 1) / n) * 360 - 180;
  const latMax = Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))) * (180 / Math.PI);
  const latMin = Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + 1)) / n))) * (180 / Math.PI);
  return { lonMin, lonMax, latMin, latMax };
}

// ---------------------------------------------------------------------------
// Binary search
// ---------------------------------------------------------------------------

function bsearch(keys: Uint32Array, target: number): number {
  let lo = 0, hi = keys.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const k = keys[mid];
    if (k === target) return mid;
    if (k < target) lo = mid + 1; else hi = mid - 1;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Colormaps
// ---------------------------------------------------------------------------

const COLORMAPS: Record<string, [number, number, number][]> = {
  rdylgn: [
    [165,   0,  38], [189,  24,  29], [213,  48,  39], [230,  82,  52],
    [245, 115,  68], [252, 152,  86], [253, 185, 110], [254, 212, 139],
    [255, 235, 171], [255, 251, 204], [235, 248, 188], [209, 238, 161],
    [169, 220, 136], [120, 198, 112], [ 75, 176,  90], [ 35, 152,  72],
    [  0, 125,  62], [  0, 104,  55], [  0,  81,  46], [  0,  68,  27],
  ],
  plasma: [
    [ 13,   8, 135], [ 70,   4, 153], [114,   1, 168], [151,   5, 175],
    [183,  29, 172], [207,  54, 160], [225,  78, 143], [238, 103, 123],
    [246, 128, 100], [251, 153,  78], [254, 177,  58], [253, 201,  38],
    [246, 224,  27], [236, 247,  28], [253, 231,  37], [246, 212,  54],
    [237, 191,  68], [224, 168,  82], [208, 145,  95], [253, 231,  37],
  ],
  viridis: [
    [ 68,   1,  84], [ 71,  22, 103], [ 72,  40, 120], [ 69,  55, 129],
    [ 63,  71, 136], [ 57,  85, 140], [ 50, 100, 142], [ 44, 114, 142],
    [ 38, 128, 142], [ 33, 143, 141], [ 30, 157, 137], [ 36, 170, 131],
    [ 55, 184, 120], [ 82, 197, 105], [116, 208,  85], [156, 218,  60],
    [196, 227,  36], [229, 234,  24], [253, 231,  37], [253, 231,  37],
  ],
};

// ---------------------------------------------------------------------------
// PNG encoder
// ---------------------------------------------------------------------------

const crcTable = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of data) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function u32be(n: number): Uint8Array {
  return new Uint8Array([(n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff]);
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const typeBytes = new TextEncoder().encode(type);
  const crcInput = new Uint8Array(4 + data.length);
  crcInput.set(typeBytes); crcInput.set(data, 4);
  const crc = crc32(crcInput);
  const chunk = new Uint8Array(4 + 4 + data.length + 4);
  chunk.set(u32be(data.length), 0); chunk.set(typeBytes, 4);
  chunk.set(data, 8); chunk.set(u32be(crc), 8 + data.length);
  return chunk;
}

async function encodePng(rgba: Uint8ClampedArray, width: number, height: number): Promise<Uint8Array> {
  const stride = width * 4;
  const raw = new Uint8Array(height * (1 + stride));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + stride)] = 0;
    raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (1 + stride) + 1);
  }
  const cs = new CompressionStream("deflate");
  const writer = cs.writable.getWriter();
  writer.write(raw); writer.close();
  const compressed = new Uint8Array(await new Response(cs.readable).arrayBuffer());
  const ihdr = new Uint8Array(13);
  ihdr.set(u32be(width), 0); ihdr.set(u32be(height), 4);
  ihdr[8] = 8; ihdr[9] = 6;
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const parts = [sig, pngChunk("IHDR", ihdr), pngChunk("IDAT", compressed), pngChunk("IEND", new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

// ---------------------------------------------------------------------------
// Render tile
// ---------------------------------------------------------------------------

const TILE_SIZE = 256;
const ALPHA = 200;

// ---------------------------------------------------------------------------
// S1 grid support
// ---------------------------------------------------------------------------

const S1_CACHE_DIR_NAME = "s1_tiles";
const s1GridCache = new Map<string, Grid>();
const s1GridLoading = new Map<string, Promise<Grid | null>>();

function s1BinPath(location: string, band: string, date: string): string {
  return join(__dirname, "..", "data", "cache", S1_CACHE_DIR_NAME, location, `${band}_${date}.bin`);
}

async function buildS1Bin(location: string, band: string, date: string, outPath: string): Promise<void> {
  console.log(`Building S1 bin: ${location}/${band}/${date}`);
  const scriptPath = join(__dirname, "s1_tile_builder.py");
  const python = pythonExe();
  const cmd = new Deno.Command(python, {
    args: [scriptPath, location, band, date, outPath],
    stdout: "inherit",
    stderr: "inherit",
  });
  const { code } = await cmd.output();
  if (code !== 0) throw new Error(`s1_tile_builder.py exited with code ${code}`);
}

export function getS1BinState(location: string, band: string, date: string): "ready" | "building" | "missing" {
  const cacheKey = `${location}/${band}/${date}`;
  if (s1GridCache.has(cacheKey)) return "ready";
  if (s1GridLoading.has(cacheKey)) return "building";
  try { Deno.statSync(s1BinPath(location, band, date)); return "ready"; } catch { /* absent */ }
  return "missing";
}

export function loadS1Grid(location: string, band: string, date: string): Promise<Grid | null> {
  const cacheKey = `${location}/${band}/${date}`;
  if (s1GridCache.has(cacheKey)) {
    const hit = s1GridCache.get(cacheKey)!;
    s1GridCache.delete(cacheKey);
    s1GridCache.set(cacheKey, hit);
    return Promise.resolve(hit);
  }
  if (s1GridLoading.has(cacheKey)) return s1GridLoading.get(cacheKey)!;

  const promise = (async () => {
    const bin = s1BinPath(location, band, date);
    let binExists = false;
    try { Deno.statSync(bin); binExists = true; } catch { /* absent */ }
    if (!binExists) await buildS1Bin(location, band, date, bin);
    const grid = await loadBin(bin);
    s1GridCache.delete(cacheKey);
    s1GridCache.set(cacheKey, grid);
    if (s1GridCache.size > GRID_CACHE_MAX) s1GridCache.delete(s1GridCache.keys().next().value!);
    console.log(`S1 grid loaded: ${cacheKey} (${grid.keys.length} pixels)`);
    return grid;
  })().finally(() => s1GridLoading.delete(cacheKey));

  s1GridLoading.set(cacheKey, promise);
  return promise;
}

// ---------------------------------------------------------------------------
// S1 location / date enumeration
// ---------------------------------------------------------------------------

const PIXELS_DIR = join(__dirname, "..", "data", "pixels");

function pythonExe(): string {
  if (Deno.env.get("PYTHON")) return Deno.env.get("PYTHON")!;
  const venvPy = join(__dirname, "..", ".venv", "bin", "python3");
  try { Deno.statSync(venvPy); return venvPy; } catch { /* no venv */ }
  return "python3";
}

export function listS1Locations(): string[] {
  const locations: string[] = [];
  try {
    for (const entry of Deno.readDirSync(PIXELS_DIR)) {
      if (!entry.isDirectory) continue;
      // Check it has at least one S1 parquet (not a .chips subdir)
      const locDir = join(PIXELS_DIR, entry.name);
      let hasParquet = false;
      try {
        for (const sub of Deno.readDirSync(locDir)) {
          if (!sub.isDirectory || !(/^\d{4}$/.test(sub.name))) continue;
          for (const f of Deno.readDirSync(join(locDir, sub.name))) {
            if (f.name.endsWith(".parquet") && !f.name.includes("coords")) {
              hasParquet = true;
              break;
            }
          }
          if (hasParquet) break;
        }
      } catch { /* unreadable */ }
      if (hasParquet) locations.push(entry.name);
    }
  } catch { /* no pixels dir */ }
  return locations.sort();
}

export async function listS1Dates(location: string): Promise<string[]> {
  // Find the first non-coords parquet in this location to sample dates from.
  // All tiles in a location share the same acquisition dates.
  const locDir = join(PIXELS_DIR, location);
  let firstParquet: string | null = null;
  try {
    for (const sub of Deno.readDirSync(locDir)) {
      if (!sub.isDirectory || !(/^\d{4}$/.test(sub.name))) continue;
      for (const f of Deno.readDirSync(join(locDir, sub.name))) {
        if (f.name.endsWith(".parquet") && !f.name.includes("coords") && !f.name.includes("by-pixel")) {
          firstParquet = join(locDir, sub.name, f.name);
          break;
        }
      }
      if (firstParquet) break;
    }
  } catch { /* unreadable */ }
  if (!firstParquet) return [];

  const python = pythonExe();
  const pq = firstParquet.replace(/\\/g, "\\\\");
  const script = [
    "import pandas as pd, json",
    `df = pd.read_parquet(r'${pq}', columns=['date','source'])`,
    "dates = sorted(str(d) for d in df[df['source']=='S1']['date'].unique())",
    "print(json.dumps(dates))",
  ].join("\n");

  const cmd = new Deno.Command(python, {
    args: ["-c", script],
    stdout: "piped",
    stderr: "inherit",
  });
  const { code, stdout } = await cmd.output();
  if (code !== 0) return [];
  return JSON.parse(new TextDecoder().decode(stdout)) as string[];
}

export async function renderTile(
  grid: Grid, z: number, x: number, y: number, cmap = "rdylgn", cutoff = 0,
): Promise<Uint8Array> {
  const stops = COLORMAPS[cmap] ?? COLORMAPS.rdylgn;
  const { lonMin: tLonMin, lonMax: tLonMax } = tileBoundsWGS84(z, x, y);
  const rgba = new Uint8ClampedArray(TILE_SIZE * TILE_SIZE * 4);
  const lonStep = (tLonMax - tLonMin) / TILE_SIZE;

  // Per-row latitude via inverse Mercator (aligns with Mercator-projected basemap).
  const tn = 2 ** z;
  const mercYTop  = Math.PI - (2 * Math.PI * y) / tn;
  const mercYStep = (2 * Math.PI) / tn / TILE_SIZE;

  const useUtm = grid.utmZone !== 0;
  const utmZoneAbs = Math.abs(grid.utmZone);
  const utmSouth   = grid.utmZone < 0;

  for (let py = 0; py < TILE_SIZE; py++) {
    const lat = Math.atan(Math.sinh(mercYTop - (py + 0.5) * mercYStep)) * (180 / Math.PI);

    for (let px = 0; px < TILE_SIZE; px++) {
      const lon = tLonMin + (px + 0.5) * lonStep;

      let xi: number, yi: number;
      if (useUtm) {
        const [E, N] = wgs84ToUtm(lat, lon, utmZoneAbs, utmSouth);
        xi = Math.round((E - grid.utmOriginE) / grid.res);
        yi = Math.round((N - grid.utmOriginN) / grid.res);
      } else {
        xi = Math.round((lon - grid.lonMin)  / grid.resX);
        yi = Math.round((grid.latMax - lat)  / grid.resY);
      }
      if (xi < 0 || xi >= grid.width || yi < 0 || yi >= grid.height) continue;

      const idx = bsearch(grid.keys, yi * grid.width + xi);
      if (idx < 0) continue;
      const prob = grid.vals[idx];
      if (prob < cutoff) continue;

      const t = Math.max(0, Math.min(1, prob)) * (stops.length - 1);
      const lo = Math.floor(t);
      const hi = Math.min(lo + 1, stops.length - 1);
      const f = t - lo;
      const [r0, g0, b0] = stops[lo];
      const [r1, g1, b1] = stops[hi];
      const i = (py * TILE_SIZE + px) * 4;
      rgba[i]     = Math.round(r0 + f * (r1 - r0));
      rgba[i + 1] = Math.round(g0 + f * (g1 - g0));
      rgba[i + 2] = Math.round(b0 + f * (b1 - b0));
      rgba[i + 3] = ALPHA;
    }
  }

  return encodePng(rgba, TILE_SIZE, TILE_SIZE);
}
