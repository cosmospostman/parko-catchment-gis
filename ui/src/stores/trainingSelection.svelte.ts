import type { BBox } from '../lib/geo.ts';

export const trainingSelection = $state<{ bbox: BBox | null; sub_role: string | null }>({ bbox: null, sub_role: null });
