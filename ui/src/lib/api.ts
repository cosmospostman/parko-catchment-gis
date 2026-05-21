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

export interface S1Status {
  state: 'missing' | 'building' | 'ready';
}

export interface HeuristicRegion {
  id: string;
  label: 'presence' | 'absence';
}

export interface NoisePixel {
  id: string;
  lat: number;
  lon: number;
  prob_woody: number | null;
}

export interface NoiseScoreResponse {
  pixels?: NoisePixel[];
  missing_data?: boolean;
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

export async function fetchS1Locations(): Promise<string[]> {
  const r = await fetch('/api/s1-locations');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function fetchS1Dates(location: string): Promise<string[]> {
  const r = await fetch(`/api/s1-dates/${location}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function fetchS1Status(location: string, band: string, date: string): Promise<S1Status> {
  const r = await fetch(`/api/s1-status/${location}/${band}/${date}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function fetchHeuristicRegions(): Promise<HeuristicRegion[]> {
  const r = await fetch('/api/heuristic-score-regions');
  if (!r.ok) return [];
  return r.json();
}

export async function fetchHeuristicScores(regionId: string): Promise<{ status: number; data: NoiseScoreResponse }> {
  const r = await fetch(`/api/heuristic-scores?region=${encodeURIComponent(regionId)}`);
  return { status: r.status, data: r.status === 200 ? await r.json() : {} };
}

export async function fetchImageryInfo(x: number, y: number, layer: string, zoom: number): Promise<ImageryInfo> {
  const r = await fetch(`/api/imagery-date?x=${x}&y=${y}&layer=${layer}&zoom=${zoom}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

