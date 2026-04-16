/**
 * tile_renderer.ts — On-the-fly ranking tile renderer.
 *
 * Loads pixel ranking CSVs into in-memory Float32Array grids, then renders
 * 256×256 RGBA PNG tiles on demand for any slippy-map z/x/y coordinate.
 *
 * No external dependencies — PNG encoding uses CompressionStream (built-in).
 */

import { join, dirname, fromFileUrl } from "jsr:@std/path";

const __dirname = dirname(fromFileUrl(import.meta.url));
const OUTPUTS_DIR = join(__dirname, "..", "outputs");

// ---------------------------------------------------------------------------
// Grid — in-memory representation of a ranking CSV
// ---------------------------------------------------------------------------

interface Grid {
  arr: Float32Array; // row-major, prob_lr values; 0 = no pixel
  lonMin: number;
  latMax: number; // top-left corner (lat decreases with row index)
  width: number;
  height: number;
  res: number; // degrees per cell
}

const gridCache = new Map<string, Grid>();

/** Scan outputs/ for *_pixel_ranking.csv files and return stem list. */
export function listRankings(): Array<{ stem: string; label: string }> {
  const results: Array<{ stem: string; label: string }> = [];
  try {
    for (const entry of Deno.readDirSync(OUTPUTS_DIR)) {
      if (!entry.isDirectory) continue;
      const dir = join(OUTPUTS_DIR, entry.name);
      for (const f of Deno.readDirSync(dir)) {
        if (f.name.endsWith("_pixel_ranking.csv")) {
          results.push({ stem: entry.name, label: entry.name });
          break;
        }
      }
    }
  } catch {
    // outputs dir may not exist yet
  }
  results.sort((a, b) => a.stem.localeCompare(b.stem));
  return results;
}

/** Find the ranking CSV path for a given stem. */
function findCsv(stem: string): string | null {
  const dir = join(OUTPUTS_DIR, stem);
  try {
    for (const f of Deno.readDirSync(dir)) {
      if (f.name.endsWith("_pixel_ranking.csv")) {
        return join(dir, f.name);
      }
    }
  } catch { /* ignore */ }
  return null;
}

/** Load (or return cached) grid for the given stem. */
export async function loadGrid(stem: string): Promise<Grid | null> {
  if (gridCache.has(stem)) return gridCache.get(stem)!;

  const csvPath = findCsv(stem);
  if (!csvPath) return null;

  console.log(`Loading ranking grid: ${csvPath}`);
  const t0 = performance.now();

  const text = await Deno.readTextFile(csvPath);
  const lines = text.split("\n");

  // Header: point_id,lon,lat,is_presence,prob_lr,rank,...
  // Find column indices from header
  const header = lines[0].split(",");
  const iLon  = header.indexOf("lon");
  const iLat  = header.indexOf("lat");
  const iProb = header.indexOf("prob_lr");
  if (iLon < 0 || iLat < 0 || iProb < 0) {
    console.error(`Missing columns in ${csvPath}`);
    return null;
  }

  const res = 0.0001;

  // First pass: find bounds
  let lonMin = Infinity, lonMax = -Infinity;
  let latMin = Infinity, latMax = -Infinity;
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const cols = lines[i].split(",");
    const lon = parseFloat(cols[iLon]);
    const lat = parseFloat(cols[iLat]);
    if (lon < lonMin) lonMin = lon;
    if (lon > lonMax) lonMax = lon;
    if (lat < latMin) latMin = lat;
    if (lat > latMax) latMax = lat;
  }

  const width  = Math.round((lonMax - lonMin) / res) + 1;
  const height = Math.round((latMax - latMin) / res) + 1;
  const arr = new Float32Array(width * height); // zeroed by default

  // Second pass: fill grid
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const cols = lines[i].split(",");
    const lon  = parseFloat(cols[iLon]);
    const lat  = parseFloat(cols[iLat]);
    const prob = parseFloat(cols[iProb]);
    if (isNaN(prob)) continue;
    const xi = Math.round((lon - lonMin) / res);
    const yi = Math.round((latMax - lat) / res);
    arr[yi * width + xi] = prob;
  }

  const grid: Grid = { arr, lonMin, latMax, width, height, res };
  gridCache.set(stem, grid);
  console.log(`Grid loaded in ${(performance.now() - t0).toFixed(0)} ms  (${width}×${height})`);
  return grid;
}

// ---------------------------------------------------------------------------
// Tile bounds — slippy tile (z, x, y) → WGS84 lon/lat bbox
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
// Colormaps — named 20-stop lookup tables
// ---------------------------------------------------------------------------

const COLORMAPS: Record<string, [number, number, number][]> = {
  rdylgn: [
    [165,   0,  38],
    [189,  24,  29],
    [213,  48,  39],
    [230,  82,  52],
    [245, 115,  68],
    [252, 152,  86],
    [253, 185, 110],
    [254, 212, 139],
    [255, 235, 171],
    [255, 251, 204],
    [235, 248, 188],
    [209, 238, 161],
    [169, 220, 136],
    [120, 198, 112],
    [ 75, 176,  90],
    [ 35, 152,  72],
    [  0, 125,  62],
    [  0, 104,  55],
    [  0,  81,  46],
    [  0,  68,  27],
  ],
  plasma: [
    [ 13,   8, 135],
    [ 70,   4, 153],
    [114,   1, 168],
    [151,   5, 175],
    [183,  29, 172],
    [207,  54, 160],
    [225,  78, 143],
    [238, 103, 123],
    [246, 128, 100],
    [251, 153,  78],
    [254, 177,  58],
    [253, 201,  38],
    [246, 224,  27],
    [236, 247,  28],
    [253, 231,  37],
    [246, 212,  54],
    [237, 191,  68],
    [224, 168,  82],
    [208, 145,  95],
    [253, 231,  37],
  ],
  viridis: [
    [ 68,   1,  84],
    [ 71,  22, 103],
    [ 72,  40, 120],
    [ 69,  55, 129],
    [ 63,  71, 136],
    [ 57,  85, 140],
    [ 50, 100, 142],
    [ 44, 114, 142],
    [ 38, 128, 142],
    [ 33, 143, 141],
    [ 30, 157, 137],
    [ 36, 170, 131],
    [ 55, 184, 120],
    [ 82, 197, 105],
    [116, 208,  85],
    [156, 218,  60],
    [196, 227,  36],
    [229, 234,  24],
    [253, 231,  37],
    [253, 231,  37],
  ],
};

// ---------------------------------------------------------------------------
// PNG encoder — minimal, no dependencies
// ---------------------------------------------------------------------------

// CRC32 lookup table
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
  crcInput.set(typeBytes);
  crcInput.set(data, 4);
  const crc = crc32(crcInput);
  const chunk = new Uint8Array(4 + 4 + data.length + 4);
  chunk.set(u32be(data.length), 0);
  chunk.set(typeBytes, 4);
  chunk.set(data, 8);
  chunk.set(u32be(crc), 8 + data.length);
  return chunk;
}

async function encodePng(rgba: Uint8ClampedArray, width: number, height: number): Promise<Uint8Array> {
  // Build raw scanlines: filter byte (0 = None) + RGBA row
  const stride = width * 4;
  const raw = new Uint8Array(height * (1 + stride));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + stride)] = 0; // filter = None
    raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (1 + stride) + 1);
  }

  // Compress with deflate (zlib wrapping needed for PNG IDAT)
  const cs = new CompressionStream("deflate");
  const writer = cs.writable.getWriter();
  writer.write(raw);
  writer.close();
  const compressed = new Uint8Array(await new Response(cs.readable).arrayBuffer());

  // IHDR: width, height, bit depth=8, color type=6 (RGBA), compression=0, filter=0, interlace=0
  const ihdr = new Uint8Array(13);
  ihdr.set(u32be(width), 0);
  ihdr.set(u32be(height), 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;

  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdrChunk = pngChunk("IHDR", ihdr);
  const idatChunk = pngChunk("IDAT", compressed);
  const iendChunk = pngChunk("IEND", new Uint8Array(0));

  const total = sig.length + ihdrChunk.length + idatChunk.length + iendChunk.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const part of [sig, ihdrChunk, idatChunk, iendChunk]) {
    out.set(part, off);
    off += part.length;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Public: render one tile
// ---------------------------------------------------------------------------

const TILE_SIZE = 256;
const ALPHA = 200; // opacity for pixels with prob > 0

export async function renderTile(
  grid: Grid,
  z: number,
  x: number,
  y: number,
  cmap = "rdylgn",
): Promise<Uint8Array> {
  const stops = COLORMAPS[cmap] ?? COLORMAPS.rdylgn;
  const { lonMin: tLonMin, lonMax: tLonMax, latMin: tLatMin, latMax: tLatMax } =
    tileBoundsWGS84(z, x, y);

  const rgba = new Uint8ClampedArray(TILE_SIZE * TILE_SIZE * 4); // zeroed = transparent

  const lonStep = (tLonMax - tLonMin) / TILE_SIZE;
  const latStep = (tLatMax - tLatMin) / TILE_SIZE;

  for (let py = 0; py < TILE_SIZE; py++) {
    // Centre of this output pixel in WGS84
    const lat = tLatMax - (py + 0.5) * latStep;
    const yi = Math.round((grid.latMax - lat) / grid.res);
    if (yi < 0 || yi >= grid.height) continue;

    for (let px = 0; px < TILE_SIZE; px++) {
      const lon = tLonMin + (px + 0.5) * lonStep;
      const xi = Math.round((lon - grid.lonMin) / grid.res);
      if (xi < 0 || xi >= grid.width) continue;

      const prob = grid.arr[yi * grid.width + xi];
      if (prob === 0) continue;

      const t = Math.max(0, Math.min(1, prob)) * (stops.length - 1);
      const lo = Math.floor(t);
      const hi = Math.min(lo + 1, stops.length - 1);
      const f = t - lo;
      const [r0, g0, b0] = stops[lo];
      const [r1, g1, b1] = stops[hi];
      const r = Math.round(r0 + f * (r1 - r0));
      const g = Math.round(g0 + f * (g1 - g0));
      const b = Math.round(b0 + f * (b1 - b0));
      const i = (py * TILE_SIZE + px) * 4;
      rgba[i]     = r;
      rgba[i + 1] = g;
      rgba[i + 2] = b;
      rgba[i + 3] = ALPHA;
    }
  }

  return encodePng(rgba, TILE_SIZE, TILE_SIZE);
}
