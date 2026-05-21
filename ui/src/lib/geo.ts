import type { LngLat } from 'maplibre-gl';

export type BBox = [number, number, number, number]; // [minLng, minLat, maxLng, maxLat]

export function parseBbox(raw: BBox | string): BBox {
  return typeof raw === 'string' ? JSON.parse(raw) : raw;
}

export function formatBbox(raw: BBox | string | undefined): string {
  if (!raw) return '—';
  const b = parseBbox(raw as BBox | string);
  return `${b[0].toFixed(5)}, ${b[1].toFixed(5)}, ${b[2].toFixed(5)}, ${b[3].toFixed(5)}`;
}

export function lngLatsToBbox(a: LngLat, b: LngLat): BBox {
  return [
    Math.min(a.lng, b.lng),
    Math.min(a.lat, b.lat),
    Math.max(a.lng, b.lng),
    Math.max(a.lat, b.lat),
  ];
}

export function bboxToFeature(bbox: BBox) {
  const [minLng, minLat, maxLng, maxLat] = bbox;
  return {
    type: 'Feature' as const,
    geometry: {
      type: 'Polygon' as const,
      coordinates: [[
        [minLng, minLat],
        [maxLng, minLat],
        [maxLng, maxLat],
        [minLng, maxLat],
        [minLng, minLat],
      ]],
    },
    properties: {},
  };
}

export function bboxToYaml(bbox: BBox): string {
  return `bbox: [${bbox.map(v => v.toFixed(6)).join(', ')}]`;
}

export function bboxPixelCount(bbox: BBox): number {
  const [minLng, minLat, maxLng, maxLat] = bbox;
  const R = 6371000;
  const dLat = (maxLat - minLat) * Math.PI / 180;
  const dLon = (maxLng - minLng) * Math.PI / 180;
  const midLat = ((minLat + maxLat) / 2) * Math.PI / 180;
  return Math.round(R * dLat * R * Math.cos(midLat) * dLon / 100);
}

export function merc(lng: number, lat: number): { x: number; y: number } {
  const x = lng * 20037508.34 / 180;
  const y = Math.log(Math.tan((90 + lat) * Math.PI / 360)) / (Math.PI / 180) * 20037508.34 / 180;
  return { x: Math.round(x), y: Math.round(y) };
}
