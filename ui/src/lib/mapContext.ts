import type { Map } from 'maplibre-gl';

export const MAP_KEY = Symbol('map');

export interface MapContext {
  getMap: () => Map | null;
}
