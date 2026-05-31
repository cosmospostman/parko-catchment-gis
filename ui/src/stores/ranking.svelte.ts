export { type Cmap, CMAP_GRADIENTS } from '../lib/colormaps.ts';

export const ranking = $state({
  location: null as string | null,
  stem: null as string | null,
  opacity: 0.6,
  cmap: 'rdylgn' as import('../lib/colormaps.ts').Cmap,
  cutoff: 0,
});
