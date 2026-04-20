import { parse as parseYaml } from "jsr:@std/yaml";
import { join, dirname, fromFileUrl } from "jsr:@std/path";
import { serveDir } from "jsr:@std/http/file-server";
import { ensureDirSync } from "jsr:@std/fs";
import { encodeHex } from "jsr:@std/encoding/hex";
import { listRankings, loadGrid, renderTile } from "./tile_renderer.ts";

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

  const upstream = await fetch(upstreamUrl, { headers: extraHeaders });
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

  // Direct-tile layers don't support ArcGIS ImageServer identify
  if (DIRECT_TILE_LAYERS.has(layer)) {
    return new Response(JSON.stringify({ date: null }), {
      headers: { "content-type": "application/json" },
    });
  }

  const geom = JSON.stringify({ x: Number(x), y: Number(y), spatialReference: { wkid: 102100 } });
  const params = new URLSearchParams({
    geometry: geom,
    geometryType: "esriGeometryPoint",
    returnGeometry: "false",
    returnCatalogItems: "true",
    catalogItemsFieldName: "capturestart,captureend,name,title",
    f: "json",
  });

  try {
    if (QGLOBE_LAYERS.has(layer)) {
      params.set("token", await getQGlobeToken());
    }
    const resp = await fetch(`${imageServerBase(layer)}?${params}`);
    const data = await resp.json() as {
      catalogItems?: { features?: Array<{ attributes: Record<string, unknown> }> };
    };

    const features = data.catalogItems?.features ?? [];
    if (features.length === 0) {
      return new Response(JSON.stringify({ date: null }), {
        headers: { "content-type": "application/json" },
      });
    }

    // Use the first (highest-priority) catalog item
    const attrs = features[0].attributes;
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
      headers: { "content-type": "application/json" },
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
      const png = await renderTile(grid, z, x, y, cmap);
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
        headers: { "content-type": "application/json" },
      });
    } catch (err) {
      console.error("Failed to load locations:", err);
      return new Response(JSON.stringify({ error: "Failed to load locations" }), {
        status: 500,
        headers: { "content-type": "application/json" },
      });
    }
  }

  if (url.pathname === "/api/sightings") {
    const sightingsPath = join(__dirname, "..", "outputs", "ala_cache", "ala_sightings.geojson");
    try {
      const data = await Deno.readFile(sightingsPath);
      return new Response(data, {
        headers: { "content-type": "application/json" },
      });
    } catch (err) {
      console.error("Failed to load sightings:", err);
      return new Response(JSON.stringify({ error: "Sightings file not found. Run: python analysis/export_sightings_geojson.py" }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });
    }
  }

  return serveDir(req, { fsRoot: join(__dirname, "public"), urlRoot: "" });
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

console.log(`parko-gis-ui running at http://localhost:${PORT}`);
Deno.serve({ port: PORT }, handler);
