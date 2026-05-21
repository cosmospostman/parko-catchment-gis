import type { NoisePixel } from '../lib/api.ts';

export const noise = $state({
  regionId: '',
  pixelData: null as NoisePixel[] | null,
  threshold: 0.5,
  pollState: 'idle' as 'idle' | 'scoring' | 'ready' | 'error',
});
