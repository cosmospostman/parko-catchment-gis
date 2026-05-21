import type { BBox } from '../lib/geo.ts';

export const trainingSelection = $state<{ bbox: BBox | null }>({ bbox: null });
