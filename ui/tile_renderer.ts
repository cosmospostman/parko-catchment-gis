/**
 * tile_renderer.ts — On-the-fly ranking tile renderer.
 *
 * Binary format (.bin): 32-byte header + N×8-byte records.
 *
 * Header (32 bytes, little-endian):
 *   f64 lonMin, f64 latMax, f64 res, u32 width, u32 height
 *
 * Records (sorted by key ascending):
 *   u32 key (= yi*width + xi), f32 prob
 *
 * The .bin is built transparently from the CSV on first request and cached on
 * disk. In memory only the typed arrays (~17 MB for 2M pixels) are held.
 */

import { join, dirname, fromFileUrl } from "jsr:@std/path";

const __dirname = dirname(fromFileUrl(import.meta.url));
const OUTPUTS_DIR = join(__dirname, "..", "outputs");

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

interface Grid {
  keys: Uint32Array;  // sorted cell indices (yi*width + xi)
  vals: Float32Array; // corresponding prob values
  lonMin: number;
  latMax: number;
  width: number;
  height: number;
  res: number;
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
// .bin build + load
// ---------------------------------------------------------------------------

const HEADER_BYTES = 32; // 3×f64 + 2×u32

function binPath(location: string, stem: string): string {
  return join(OUTPUTS_DIR, location, `${stem}.bin`);
}

async function buildBin(csvPath: string, outPath: string): Promise<void> {
  console.log(`Building binary cache: ${outPath}`);
  const t0 = performance.now();
  const res = 0.0001;

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

  // Pass 0: header
  let iLon = -1, iLat = -1, iProb = -1;
  await streamLines((cols, lineNo) => {
    if (lineNo !== 0) return;
    iLon  = cols.indexOf("lon");
    iLat  = cols.indexOf("lat");
    iProb = cols.findIndex((c) => c.startsWith("prob_"));
  });
  if (iLon < 0 || iLat < 0 || iProb < 0) throw new Error(`Missing columns in ${csvPath}`);

  // Pass 1: bounds
  let lonMin = Infinity, lonMax = -Infinity, latMin = Infinity, latMax = -Infinity;
  await streamLines((cols, lineNo) => {
    if (lineNo === 0) return;
    const lon = parseFloat(cols[iLon]);
    const lat = parseFloat(cols[iLat]);
    if (lon < lonMin) lonMin = lon;
    if (lon > lonMax) lonMax = lon;
    if (lat < latMin) latMin = lat;
    if (lat > latMax) latMax = lat;
  });

  const width  = Math.round((lonMax - lonMin) / res) + 1;
  const height = Math.round((latMax - latMin) / res) + 1;

  // Pass 2: collect records
  const records: Array<[number, number]> = [];
  await streamLines((cols, lineNo) => {
    if (lineNo === 0) return;
    const lon  = parseFloat(cols[iLon]);
    const lat  = parseFloat(cols[iLat]);
    const prob = parseFloat(cols[iProb]);
    if (isNaN(prob)) return;
    const xi = Math.round((lon - lonMin) / res);
    const yi = Math.round((latMax - lat) / res);
    records.push([yi * width + xi, prob]);
  });

  records.sort((a, b) => a[0] - b[0]);

  // Write binary file
  const n = records.length;
  const buf = new ArrayBuffer(HEADER_BYTES + n * 8);
  const dv  = new DataView(buf);
  dv.setFloat64(0,  lonMin, true);
  dv.setFloat64(8,  latMax, true);
  dv.setFloat64(16, res,    true);
  dv.setUint32(24,  width,  true);
  dv.setUint32(28,  height, true);
  let off = HEADER_BYTES;
  for (const [key, prob] of records) {
    dv.setUint32(off,     key,  true);
    dv.setFloat32(off + 4, prob, true);
    off += 8;
  }
  await Deno.writeFile(outPath, new Uint8Array(buf));
  console.log(`Binary cache built in ${(performance.now() - t0).toFixed(0)} ms  (${n} pixels, ${width}×${height})`);
}

async function loadBin(path: string): Promise<Grid> {
  const raw = await Deno.readFile(path);
  const dv  = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const lonMin  = dv.getFloat64(0,  true);
  const latMax  = dv.getFloat64(8,  true);
  const res     = dv.getFloat64(16, true);
  const width   = dv.getUint32(24,  true);
  const height  = dv.getUint32(28,  true);
  const n = (raw.byteLength - HEADER_BYTES) / 8;
  const keys = new Uint32Array(n);
  const vals = new Float32Array(n);
  let off = HEADER_BYTES;
  for (let i = 0; i < n; i++) {
    keys[i] = dv.getUint32(off,     true);
    vals[i] = dv.getFloat32(off + 4, true);
    off += 8;
  }
  return { keys, vals, lonMin, latMax, width, height, res };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function listRankings(): Record<string, Array<{ stem: string; label: string }>> {
  const results: Record<string, Array<{ stem: string; label: string }>> = {};
  try {
    for (const entry of Deno.readDirSync(OUTPUTS_DIR)) {
      if (!entry.isDirectory) continue;
      const dir = join(OUTPUTS_DIR, entry.name);
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
    const csv = join(OUTPUTS_DIR, location, `${stem}.csv`);

    let binExists = false;
    try { Deno.statSync(bin); binExists = true; } catch { /* absent */ }
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

export async function renderTile(
  grid: Grid, z: number, x: number, y: number, cmap = "rdylgn", cutoff = 0,
): Promise<Uint8Array> {
  const stops = COLORMAPS[cmap] ?? COLORMAPS.rdylgn;
  const { lonMin: tLonMin, lonMax: tLonMax, latMax: tLatMax } = tileBoundsWGS84(z, x, y);
  const rgba = new Uint8ClampedArray(TILE_SIZE * TILE_SIZE * 4);
  const lonStep = (tLonMax - tLonMin) / TILE_SIZE;
  const latStep = (tLatMax - tileBoundsWGS84(z, x, y).latMin) / TILE_SIZE;

  for (let py = 0; py < TILE_SIZE; py++) {
    const lat = tLatMax - (py + 0.5) * latStep;
    const yi = Math.round((grid.latMax - lat) / grid.res);
    if (yi < 0 || yi >= grid.height) continue;

    for (let px = 0; px < TILE_SIZE; px++) {
      const lon = tLonMin + (px + 0.5) * lonStep;
      const xi = Math.round((lon - grid.lonMin) / grid.res);
      if (xi < 0 || xi >= grid.width) continue;

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
