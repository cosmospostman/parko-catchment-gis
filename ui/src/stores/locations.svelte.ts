import type { LocationsGeoJSON, RankingsMap } from '../lib/api.ts';

export const locationsStore = $state({
  geojson: null as LocationsGeoJSON | null,
  rankings: {} as RankingsMap,
  mapReady: false,
  s2tilesReady: false,
});
