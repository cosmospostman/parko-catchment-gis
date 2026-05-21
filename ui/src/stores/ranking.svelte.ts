export type Cmap = 'rdylgn' | 'plasma' | 'viridis';

export const CMAP_GRADIENTS: Record<Cmap, string> = {
  rdylgn:  'linear-gradient(to right, #a50026, #f46d43, #fee08b, #d9ef8b, #006837)',
  plasma:  'linear-gradient(to right, #0d0887, #cc4778, #f89540, #f0f921)',
  viridis: 'linear-gradient(to right, #440154, #31688e, #35b779, #fde725)',
};

export const ranking = $state({
  location: null as string | null,
  stem: null as string | null,
  opacity: 0.6,
  cmap: 'rdylgn' as Cmap,
  cutoff: 0,
});
