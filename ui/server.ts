import { parse as parseYaml } from "jsr:@std/yaml";
import { join, dirname, fromFileUrl } from "jsr:@std/path";
import { serveDir } from "jsr:@std/http/file-server";
import { ensureDirSync } from "jsr:@std/fs";
import { encodeHex } from "jsr:@std/encoding/hex";
import { listRankings, loadGrid, renderTile, listS1Locations, listS1Dates, loadS1Grid, getS1BinState } from "./tile_renderer.ts";

const __dirname = dirname(fromFileUrl(import.meta.url));
const LOCATIONS_DIR = join(__dirname, "..", "data", "locations");
const WMS_CACHE_DIR = join(__dirname, "..", "data", "cache", "wms");
const PORT = Number(Deno.env.get("PORT") ?? 3000);

ensureDirSync(WMS_CACHE_DIR);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface SubBbox {
  label?: string;
  role?: string;
  bbox: [number, number, number, number];
}

interface LocationYaml {
  name?: string;
  bbox?: [number, number, number, number];
  score_bbox?: [number, number, number, number];
  notes?: string;
  sub_bboxes?: Record<string, SubBbox>;
}

interface TrainingRegion {
  id: string;
  name: string;
  label: string;   // "presence" | "absence"
  bbox: [number, number, number, number];
  year?: number;
  notes?: string;
}

interface TrainingYaml {
  regions: TrainingRegion[];
}

// ---------------------------------------------------------------------------
// YAML → GeoJSON
// ---------------------------------------------------------------------------

function bboxToPolygon(bbox: [number, number, number, number]): GeoJSON.Polygon {
  const [lon_min, lat_min, lon_max, lat_max] = bbox;
  return {
    type: "Polygon",
    coordinates: [[
      [lon_min, lat_min],
      [lon_max, lat_min],
      [lon_max, lat_max],
      [lon_min, lat_max],
      [lon_min, lat_min],
    ]],
  };
}

function loadTrainingRegions(features: GeoJSON.Feature[]): void {
  const path = join(LOCATIONS_DIR, "training.yaml");
  try {
    const raw = Deno.readTextFileSync(path);
    const data = parseYaml(raw) as TrainingYaml;
    for (const region of data.regions ?? []) {
      if (!region.bbox) continue;
      features.push({
        type: "Feature",
        geometry: bboxToPolygon(region.bbox),
        properties: {
          id: region.id,
          name: region.name,
          label: region.name,
          role: "sub_bbox",
          sub_role: region.label,   // "presence" | "absence"
          parent_id: "training",
          year: region.year ?? null,
          notes: region.notes ?? null,
          bbox: region.bbox,
        },
      });
    }
  } catch (_) {
    // training.yaml absent or unreadable — skip silently
  }
}

function loadLocations(): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = [];

  loadTrainingRegions(features);

  for (const entry of Deno.readDirSync(LOCATIONS_DIR)) {
    if (!entry.name.endsWith(".yaml")) continue;
    if (entry.name === "training.yaml") continue;  // handled above
    const slug = entry.name.replace(/\.yaml$/, "");
    const raw = Deno.readTextFileSync(join(LOCATIONS_DIR, entry.name));
    const loc = parseYaml(raw) as LocationYaml;

    if (!loc?.bbox) continue;

    features.push({
      type: "Feature",
      geometry: bboxToPolygon(loc.bbox),
      properties: {
        id: slug,
        name: loc.name ?? slug,
        notes: loc.notes ?? null,
        role: "location",
        bbox: loc.bbox,
      },
    });

    if (loc.score_bbox) {
      features.push({
        type: "Feature",
        geometry: bboxToPolygon(loc.score_bbox),
        properties: {
          id: `${slug}__score_bbox`,
          parent_id: slug,
          label: "Score bbox",
          role: "score_bbox",
          sub_role: null,
          bbox: loc.score_bbox,
        },
      });
    }

    if (loc.sub_bboxes) {
      for (const [key, sub] of Object.entries(loc.sub_bboxes)) {
        if (!sub.bbox) continue;
        features.push({
          type: "Feature",
          geometry: bboxToPolygon(sub.bbox),
          properties: {
            id: `${slug}__${key}`,
            parent_id: slug,
            label: sub.label ?? key,
            role: "sub_bbox",
            sub_role: sub.role ?? null,
            bbox: sub.bbox,
          },
        });
      }
    }
  }

  return { type: "FeatureCollection", features };
}

// ---------------------------------------------------------------------------
// WMS tile proxy with disk cache
// ---------------------------------------------------------------------------

const PUBLIC_LAYERS = new Set([
  "LatestStateProgram_AllUsers",
  "LatestSatelliteWOS_AllUsers",
]);

const QGLOBE_LAYERS = new Set([
  "LatestStateProgram_QGovSISPUsers",
]);

// Extra direct-tile layers (no proxy — served straight from the client)
// Listed here so the server knows they're valid for /api/imagery-date (returns null)
const DIRECT_TILE_LAYERS = new Set([
  "EsriWorldImagery",
]);

const SPATIAL_IMG_BASE = "https://spatial-img.information.qld.gov.au";
const QGLOBE_URL = "https://qldglobe.information.qld.gov.au";

function wmsBase(layer: string): string {
  return `${SPATIAL_IMG_BASE}/arcgis/services/Basemaps/${layer}/ImageServer/WMSServer`;
}

function tileBase(layer: string): string {
  return `${SPATIAL_IMG_BASE}/arcgis/rest/services/Basemaps/${layer}/ImageServer/tile`;
}

function imageServerBase(layer: string): string {
  return `${SPATIAL_IMG_BASE}/arcgis/rest/services/Basemaps/${layer}/ImageServer/identify`;
}

// ---------------------------------------------------------------------------
// Queensland Globe token manager
// ---------------------------------------------------------------------------

interface QGlobeToken {
  token: string;
  expires: number; // ms epoch
}

let qglobeToken: QGlobeToken | null = null;

async function fetchQGlobeToken(): Promise<QGlobeToken> {
  // Step 1: get session cookie + XSRF token
  const homeResp = await fetch(QGLOBE_URL + "/", { redirect: "follow" });
  const setCookies = homeResp.headers.getSetCookie();

  let sessionCookie = "";
  let xsrfToken = "";
  for (const c of setCookies) {
    const [pair] = c.split(";");
    if (pair.startsWith("this.sid=")) sessionCookie = pair;
    if (pair.startsWith("XSRF-TOKEN=")) {
      sessionCookie += (sessionCookie ? "; " : "") + pair;
      xsrfToken = pair.replace("XSRF-TOKEN=", "");
    }
  }

  // Step 2: POST to token endpoint
  const tokenResp = await fetch(QGLOBE_URL + "/api/qldglobe/public/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Origin": QGLOBE_URL,
      "Referer": QGLOBE_URL + "/",
      "X-XSRF-TOKEN": xsrfToken,
      "X-Requested-With": "XMLHttpRequest",
      "Cookie": sessionCookie,
    },
    body: JSON.stringify({
      url: `${SPATIAL_IMG_BASE}/arcgis/rest/services/Basemaps/LatestStateProgram_QGovSISPUsers/ImageServer`,
      location: {
        href: QGLOBE_URL + "/",
        hostname: "qldglobe.information.qld.gov.au",
        pathname: "/",
        protocol: "https:",
      },
    }),
  });

  if (!tokenResp.ok) throw new Error(`Token fetch failed: ${tokenResp.status}`);
  const data = await tokenResp.json() as { token: string; expires: number };
  console.log(`QGlobe token obtained, expires ${new Date(data.expires).toISOString()}`);
  return { token: data.token, expires: data.expires };
}

async function getQGlobeToken(): Promise<string> {
  // Refresh if missing or expiring within 60 seconds
  if (!qglobeToken || Date.now() > qglobeToken.expires - 60_000) {
    qglobeToken = await fetchQGlobeToken();
  }
  return qglobeToken.token;
}

// Kick off initial token fetch in background
fetchQGlobeToken().then(t => { qglobeToken = t; }).catch(e => {
  console.warn("Initial QGlobe token fetch failed:", e);
});

async function cacheKey(layer: string, params: URLSearchParams): Promise<string> {
  // Include layer name so different services don't collide
  const canonical = layer + "?" + new URLSearchParams([...params.entries()].sort()).toString();
  const buf = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(canonical));
  return encodeHex(new Uint8Array(buf));
}

async function fetchAndCache(upstreamUrl: string, cachePath: string, extraHeaders?: Record<string, string>): Promise<Response> {
  // Cache hit
  try {
    const cached = await Deno.readFile(cachePath);
    return new Response(cached, {
      headers: {
        "content-type": "image/jpeg",
        "x-tile-cache": "HIT",
        "cache-control": "public, max-age=86400",
      },
    });
  } catch {
    // Cache miss — fall through
  }

  const upstream = await fetch(upstreamUrl, { headers: extraHeaders, signal: AbortSignal.timeout(8000) });
  if (!upstream.ok || !upstream.body) {
    if (upstream.status === 404) return new Response(null, { status: 204 });
    console.error(`Upstream tile error ${upstream.status} for ${upstreamUrl.split("?")[0]}`);
    return new Response("Upstream error", { status: 502 });
  }

  const bytes = new Uint8Array(await upstream.arrayBuffer());
  Deno.writeFile(cachePath, bytes).catch((e) =>
    console.warn("Tile cache write failed:", e)
  );

  return new Response(bytes, {
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "image/jpeg",
      "x-tile-cache": "MISS",
      "cache-control": "public, max-age=86400",
    },
  });
}

async function handleWmsTile(req: Request, layer: string): Promise<Response> {
  const params = new URL(req.url).searchParams;
  const key = await cacheKey(layer, params);
  const cachePath = join(WMS_CACHE_DIR, key + ".jpg");

  const upstreamUrl = new URL(wmsBase(layer));
  upstreamUrl.search = params.toString();
  return fetchAndCache(upstreamUrl.toString(), cachePath);
}

async function handleRestTile(layer: string, z: string, x: string, y: string): Promise<Response> {
  const cacheInput = `${layer}/tile/${z}/${x}/${y}`;
  const buf = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(cacheInput));
  const key = encodeHex(new Uint8Array(buf));
  const cachePath = join(WMS_CACHE_DIR, key + ".jpg");
  // Serve from cache without needing a token
  try {
    const cached = await Deno.readFile(cachePath);
    return new Response(cached, {
      headers: { "content-type": "image/jpeg", "x-tile-cache": "HIT", "cache-control": "public, max-age=86400" },
    });
  } catch { /* cache miss */ }
  const token = await getQGlobeToken();
  const upstreamUrl = `${tileBase(layer)}/${z}/${y}/${x}?token=${token}`;
  return fetchAndCache(upstreamUrl, cachePath);
}

// ---------------------------------------------------------------------------
// ImageServer identify — capture date for a point
// ---------------------------------------------------------------------------

async function handleImageryDate(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const x = url.searchParams.get("x");
  const y = url.searchParams.get("y");
  const layer = url.searchParams.get("layer") ?? "LatestStateProgram_QGovSISPUsers";
  if (!x || !y) {
    return new Response(JSON.stringify({ error: "x and y required" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }

  // Esri World Imagery — query the public Citations layer for capture metadata
  if (layer === "EsriWorldImagery") {
    const zoom = Number(url.searchParams.get("zoom") ?? "14");
    const nx = Number(x), ny = Number(y);
    const esriIdentifyUrl = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/identify";
    const params = new URLSearchParams({
      geometry: JSON.stringify({ x: nx, y: ny, spatialReference: { wkid: 102100 } }),
      geometryType: "esriGeometryPoint",
      layers: "all:4",
      tolerance: "0",
      mapExtent: `${nx - 2000},${ny - 2000},${nx + 2000},${ny + 2000}`,
      imageDisplay: "256,256,96",
      returnGeometry: "false",
      f: "json",
    });
    try {
      const data = await fetch(`${esriIdentifyUrl}?${params}`, { signal: AbortSignal.timeout(8000) }).then(r => r.json());
      type EsriResult = { attributes: Record<string, string> };
      const results: EsriResult[] = data.results ?? [];
      // Pick the citation matching current zoom level
      const match = results.find(r => {
        const from = parseInt(r.attributes["FROM_CACHE_LEVEL"]);
        const to   = parseInt(r.attributes["TO_CACHE_LEVEL"]);
        return !isNaN(from) && !isNaN(to) && from <= zoom && zoom <= to;
      }) ?? results[0];
      if (!match) {
        return new Response(JSON.stringify({ date: null }), {
          headers: { "content-type": "application/json" },
        });
      }
      const a = match.attributes;
      const raw = a["DATE (YYYYMMDD)"];
      const dateStr = raw && raw !== "Null" && raw.length === 8
        ? `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`
        : null;
      return new Response(JSON.stringify({
        capturestart: dateStr,
        captureend:   dateStr,
        name:  a["SOURCE_INFO"] ?? null,
        title: `${a["SOURCE_INFO"] ?? ""} ${a["RESOLUTION (M)"] ? `(${a["RESOLUTION (M)"]}m)` : ""}`.trim() || null,
      }), {
        headers: { "content-type": "application/json", "cache-control": "public, max-age=3600" },
      });
    } catch (err) {
      console.error("Esri Citations identify failed:", err);
      return new Response(JSON.stringify({ date: null }), {
        headers: { "content-type": "application/json" },
      });
    }
  }

  // Other direct-tile layers don't support identify
  if (DIRECT_TILE_LAYERS.has(layer)) {
    return new Response(JSON.stringify({ date: null }), {
      headers: { "content-type": "application/json" },
    });
  }

  // QGovSISP layer doesn't support identify — fall back to the public equivalent
  const identifyLayer = layer === "LatestStateProgram_QGovSISPUsers"
    ? "LatestStateProgram_AllUsers"
    : layer;

  const nx = Number(x);
  const ny = Number(y);
  // Sample a small cluster of points — mosaic catalog footprints can be tighter than
  // the rendered extent, so probing nearby points surfaces high-res items at seam edges.
  const offsets = [0, 2000, -2000];
  const probePoints = offsets.flatMap(dx => offsets.map(dy => ({ x: nx + dx, y: ny + dy })));

  const makeParams = (px: number, py: number) => new URLSearchParams({
    geometry: JSON.stringify({ x: px, y: py, spatialReference: { wkid: 102100 } }),
    geometryType: "esriGeometryPoint",
    returnGeometry: "false",
    returnCatalogItems: "true",
    catalogItemsFieldName: "capturestart,captureend,name,title,lowps",
    f: "json",
  });

  try {
    const responses = await Promise.all(
      probePoints.map(p => fetch(`${imageServerBase(identifyLayer)}?${makeParams(p.x, p.y)}`, { signal: AbortSignal.timeout(8000) }).then(r => r.ok ? r.json() : null).catch(() => null))
    );

    type Feature = { attributes: Record<string, unknown> };
    const allFeatures: Feature[] = [];
    const seen = new Set<unknown>();
    for (const data of responses) {
      if (!data) continue;
      for (const f of (data.catalogItems?.features ?? []) as Feature[]) {
        const id = f.attributes.objectid ?? f.attributes.name;
        if (!seen.has(id)) { seen.add(id); allFeatures.push(f); }
      }
    }

    if (allFeatures.length === 0) {
      return new Response(JSON.stringify({ date: null }), {
        headers: { "content-type": "application/json" },
      });
    }

    // Pick the item with the finest native resolution (smallest lowps)
    allFeatures.sort((a, b) => ((a.attributes.lowps as number) ?? 9999) - ((b.attributes.lowps as number) ?? 9999));
    const attrs = allFeatures[0].attributes;
    const start = attrs.capturestart ? new Date(attrs.capturestart as number) : null;
    const end   = attrs.captureend   ? new Date(attrs.captureend   as number) : null;

    return new Response(JSON.stringify({
      capturestart: start?.toISOString().slice(0, 10) ?? null,
      captureend:   end?.toISOString().slice(0, 10)   ?? null,
      name:  attrs.name  ?? null,
      title: attrs.title ?? null,
    }), {
      headers: {
        "content-type": "application/json",
        "cache-control": "public, max-age=3600",
      },
    });
  } catch (err) {
    console.error("ImageServer identify failed:", err);
    return new Response(JSON.stringify({ error: "upstream error" }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);

  // /tile/<layerName>/<z>/<x>/<y> — REST tile proxy for QGlobe authenticated layers
  const tileMatch = url.pathname.match(/^\/tile\/([^/]+)\/(\d+)\/(\d+)\/(\d+)$/);
  if (tileMatch) {
    const [, layer, z, x, y] = tileMatch;
    if (!QGLOBE_LAYERS.has(layer)) {
      return new Response("Unknown layer", { status: 400 });
    }
    return handleRestTile(layer, z, x, y);
  }

  // /wms/<layerName>?... — WMS proxy for public layers
  const wmsMatch = url.pathname.match(/^\/wms\/([^/]+)$/);
  if (wmsMatch) {
    const layer = wmsMatch[1];
    if (!PUBLIC_LAYERS.has(layer)) {
      return new Response("Unknown layer", { status: 400 });
    }
    return handleWmsTile(req, layer);
  }

  if (url.pathname === "/api/imagery-date") {
    return handleImageryDate(req);
  }

  if (url.pathname === "/api/rankings") {
    return new Response(JSON.stringify(listRankings()), {
      headers: { "content-type": "application/json", "cache-control": "no-store" },
    });
  }

  const rankingTileMatch = url.pathname.match(/^\/ranking-tile\/([^/]+)\/([^/]+)\/(\d+)\/(\d+)\/(\d+)$/);
  if (rankingTileMatch) {
    const [, location, stem, zs, xs, ys] = rankingTileMatch;
    const z = parseInt(zs), x = parseInt(xs), y = parseInt(ys);
    try {
      const grid = await loadGrid(location, stem);
      if (!grid) return new Response("Unknown stem", { status: 404 });
      const cmap = url.searchParams.get("cmap") ?? "rdylgn";
      const cutoff = parseFloat(url.searchParams.get("cutoff") ?? "0");
      const png = await renderTile(grid, z, x, y, cmap, cutoff);
      return new Response(png as unknown as BodyInit, {
        headers: {
          "content-type": "image/png",
          "cache-control": "public, max-age=3600",
        },
      });
    } catch (err) {
      console.error("Tile render error:", err);
      return new Response("Render error", { status: 500 });
    }
  }

  if (url.pathname === "/api/locations") {
    try {
      const geojson = loadLocations();
      return new Response(JSON.stringify(geojson), {
        headers: { "content-type": "application/json", "cache-control": "public, max-age=300" },
      });
    } catch (err) {
      console.error("Failed to load locations:", err);
      return new Response(JSON.stringify({ error: "Failed to load locations" }), {
        status: 500,
        headers: { "content-type": "application/json" },
      });
    }
  }

  if (url.pathname === "/api/catchments") {
    try {
      const catchmentsDir = join(__dirname, "..", "data", "catchments");
      const features: GeoJSON.Feature[] = [];
      for (const entry of Deno.readDirSync(catchmentsDir)) {
        if (!entry.name.endsWith(".geojson")) continue;
        const raw = Deno.readTextFileSync(join(catchmentsDir, entry.name));
        const parsed = JSON.parse(raw);
        if (parsed.type === "FeatureCollection") {
          features.push(...parsed.features);
        } else {
          features.push(parsed as GeoJSON.Feature);
        }
      }
      const fc: GeoJSON.FeatureCollection = { type: "FeatureCollection", features };
      return new Response(JSON.stringify(fc), {
        headers: { "content-type": "application/json", "cache-control": "public, max-age=300" },
      });
    } catch (err) {
      console.error("Failed to load catchments:", err);
      return new Response(JSON.stringify({ error: "Failed to load catchments" }), {
        status: 500,
        headers: { "content-type": "application/json" },
      });
    }
  }

  const s1StatusMatch = url.pathname.match(/^\/api\/s1-status\/([^/]+)\/([^/]+)\/([^/]+)$/);
  if (s1StatusMatch) {
    const [, location, band, date] = s1StatusMatch;
    let state = getS1BinState(location, band, date);
    if (state === "missing") {
      // Kick off the build in the background; immediately return "building"
      loadS1Grid(location, band, date).catch(err => console.error("S1 build error:", err));
      state = "building";
    }
    return new Response(JSON.stringify({ state }), {
      headers: { "content-type": "application/json", "cache-control": "no-store" },
    });
  }

  if (url.pathname === "/api/s1-locations") {
    return new Response(JSON.stringify(listS1Locations()), {
      headers: { "content-type": "application/json", "cache-control": "public, max-age=60" },
    });
  }

  const s1DatesMatch = url.pathname.match(/^\/api\/s1-dates\/([^/]+)$/);
  if (s1DatesMatch) {
    const location = s1DatesMatch[1];
    try {
      const dates = await listS1Dates(location);
      return new Response(JSON.stringify(dates), {
        headers: { "content-type": "application/json", "cache-control": "public, max-age=60" },
      });
    } catch (err) {
      console.error("listS1Dates failed:", err);
      return new Response(JSON.stringify([]), {
        headers: { "content-type": "application/json" },
      });
    }
  }

  const s1TileMatch = url.pathname.match(/^\/s1-tile\/([^/]+)\/([^/]+)\/([^/]+)\/(\d+)\/(\d+)\/(\d+)$/);
  if (s1TileMatch) {
    const [, location, band, date, zs, xs, ys] = s1TileMatch;
    if (!["vh", "vv"].includes(band)) return new Response("Unknown band", { status: 400 });
    const z = parseInt(zs), x = parseInt(xs), y = parseInt(ys);
    try {
      const grid = await loadS1Grid(location, band, date);
      if (!grid) return new Response("No data", { status: 404 });
      const cmap = url.searchParams.get("cmap") ?? "plasma";
      const png = await renderTile(grid, z, x, y, cmap, 0);
      return new Response(png as unknown as BodyInit, {
        headers: { "content-type": "image/png", "cache-control": "public, max-age=3600" },
      });
    } catch (err) {
      console.error("S1 tile render error:", err);
      return new Response("Render error", { status: 500 });
    }
  }

  if (url.pathname === "/api/sightings") {
    const sightingsPath = join(__dirname, "..", "outputs", "ala_cache", "ala_sightings.geojson");
    try {
      const data = await Deno.readFile(sightingsPath);
      return new Response(data, {
        headers: { "content-type": "application/json", "cache-control": "public, max-age=3600" },
      });
    } catch (err) {
      console.error("Failed to load sightings:", err);
      return new Response(JSON.stringify({ error: "Sightings file not found. Run: python analysis/export_sightings_geojson.py" }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    }
  }

  const staticRes = await serveDir(req, { fsRoot: join(__dirname, "public"), urlRoot: "" });
  const path = new URL(req.url).pathname;
  if (path === "/" || path === "/index.html" || path === "/app.js") {
    staticRes.headers.set("cache-control", "no-cache");
  }
  return staticRes;
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

console.log(`parko-gis-ui running at http://localhost:${PORT}`);
Deno.serve({ port: PORT }, handler);
