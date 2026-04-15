import { parse as parseYaml } from "jsr:@std/yaml";
import { join, dirname, fromFileUrl } from "jsr:@std/path";
import { serveDir } from "jsr:@std/http/file-server";
import { ensureDirSync } from "jsr:@std/fs";
import { encodeHex } from "jsr:@std/encoding/hex";

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
  notes?: string;
  sub_bboxes?: Record<string, SubBbox>;
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

function loadLocations(): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = [];

  for (const entry of Deno.readDirSync(LOCATIONS_DIR)) {
    if (!entry.name.endsWith(".yaml")) continue;
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

const WMS_BASE =
  "https://spatial-img.information.qld.gov.au/arcgis/services/Basemaps/" +
  "LatestStateProgram_AllUsers/ImageServer/WMSServer";

async function cacheKey(params: URLSearchParams): Promise<string> {
  // Normalise param order so the same tile always maps to the same key
  const canonical = new URLSearchParams([...params.entries()].sort()).toString();
  const buf = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(canonical));
  return encodeHex(new Uint8Array(buf));
}

async function handleWmsTile(req: Request): Promise<Response> {
  const inUrl = new URL(req.url);
  const params = inUrl.searchParams;

  const key = await cacheKey(params);
  const cachePath = join(WMS_CACHE_DIR, key + ".jpg");

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

  // Fetch from upstream
  const upstreamUrl = new URL(WMS_BASE);
  upstreamUrl.search = params.toString();

  const upstream = await fetch(upstreamUrl.toString());
  if (!upstream.ok || !upstream.body) {
    return new Response("WMS upstream error", { status: 502 });
  }

  const bytes = new Uint8Array(await upstream.arrayBuffer());

  // Write to cache (best-effort; don't fail the request if write fails)
  Deno.writeFile(cachePath, bytes).catch((e) =>
    console.warn("WMS cache write failed:", e)
  );

  return new Response(bytes, {
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "image/jpeg",
      "x-tile-cache": "MISS",
      "cache-control": "public, max-age=86400",
    },
  });
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

function handler(req: Request): Response | Promise<Response> {
  const url = new URL(req.url);

  if (url.pathname === "/wms") {
    return handleWmsTile(req);
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

  return serveDir(req, { fsRoot: join(__dirname, "public"), urlRoot: "" });
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

console.log(`parko-gis-ui running at http://localhost:${PORT}`);
Deno.serve({ port: PORT }, handler);
