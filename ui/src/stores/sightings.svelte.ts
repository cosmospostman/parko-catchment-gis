import type { Feature, Point } from 'geojson';
import type { SightingProperties } from '../lib/api.ts';

export const sightings = $state({
  features: [] as Feature<Point, SightingProperties>[],
  totalCount: 0,
  yearMin: 1900,
  yearMax: 2026,
});
