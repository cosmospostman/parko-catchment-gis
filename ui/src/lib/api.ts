import type { FeatureCollection, Feature, Polygon } from 'geojson';

export interface LocationProperties {
  id: string;
  name: string;
  role: 'location' | 'sub_bbox' | 'score_bbox';
  sub_role?: 'presence' | 'absence' | 'survey';
  parent_id?: string;
  bbox?: [number, number, number, number] | string;
  notes?: string;
  year?: number;
  label?: string;
}

export type LocationFeature = Feature<Polygon, LocationProperties>;
export type LocationsGeoJSON = FeatureCollection<Polygon, LocationProperties>;

export interface RankingRun {
  stem: string;
  label: string;
}

export type RankingsMap = Record<string, RankingRun[]>;

export interface SightingProperties {
  eventDate?: string;
  year?: number;
  month?: number;
  recordedBy?: string;
  dataResourceName?: string;
  basisOfRecord?: string;
  coordinateUncertaintyInMeters?: number;
  spatiallyValid?: boolean | string;
}

export interface HeuristicRegion {
  id: string;
  label: 'presence' | 'absence';
}

export interface ImageryInfo {
  capturestart?: string;
  captureend?: string;
  name?: string;
}

export async function fetchLocations(): Promise<LocationsGeoJSON> {
  const r = await fetch('/api/locations');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function fetchRankings(): Promise<RankingsMap> {
  const r = await fetch('/api/rankings');
  if (!r.ok) return {};
  return r.json();
}

export async function fetchCatchments(): Promise<FeatureCollection> {
  const r = await fetch('/api/catchments');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function fetchSightings(): Promise<FeatureCollection<GeoJSON.Point, SightingProperties>> {
  const r = await fetch('/api/sightings');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function fetchHeuristicRegions(): Promise<HeuristicRegion[]> {
  const r = await fetch('/api/heuristic-score-regions');
  if (!r.ok) return [];
  return r.json();
}

export interface ChunkCoverage {
  years: number[];
  tiles: Record<number, string[]>;
}

export async function fetchChunkCoverage(coords: [number, number, number, number]): Promise<ChunkCoverage> {
  const bbox = coords.join(',');
  const r = await fetch(`/api/chunk-coverage?bbox=${bbox}`);
  if (!r.ok) return { years: [], tiles: {} };
  const data = await r.json() as ChunkCoverage;
  return { years: data.years ?? [], tiles: data.tiles ?? {} };
}

export interface TimeseriesPoint {
  date: string;
  ndvi: number | null;
  ndvi_p25: number | null;
  ndvi_p75: number | null;
  mavi: number | null;
  mavi_p25: number | null;
  mavi_p75: number | null;
  vh_vv: number | null;
  vh_vv_p25: number | null;
  vh_vv_p75: number | null;
}

export interface PixelTimeseries {
  year: number;
  tile: string;
  series: TimeseriesPoint[];
}

export async function fetchPixelTimeseries(
  coords: [number, number, number, number],
  year: number,
  tile: string,
): Promise<PixelTimeseries | null> {
  const bbox = coords.join(',');
  const r = await fetch(`/api/pixel-timeseries?bbox=${bbox}&year=${year}&tile=${encodeURIComponent(tile)}`);
  if (!r.ok) return null;
  return r.json() as Promise<PixelTimeseries>;
}

export async function fetchImageryInfo(x: number, y: number, layer: string, zoom: number): Promise<ImageryInfo> {
  const r = await fetch(`/api/imagery-date?x=${x}&y=${y}&layer=${layer}&zoom=${zoom}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

