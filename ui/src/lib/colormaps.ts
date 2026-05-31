/**
 * colormaps.ts — MapLibre raster-color expressions for TAM score visualization.
 *
 * Scores are stored as uint8 R-channel values (0 = no data, 1–100 = prob_tam%).
 * The MapLibre raster-color-mix extracts the R channel, giving 0–255.
 * We rescale 0–255 → 0–100 in the expression, then apply the colormap.
 */

export type Cmap = 'rdylgn' | 'plasma' | 'viridis';

// Stop arrays: [input_value_0_to_100, r, g, b] stops for interpolation
const STOPS: Record<Cmap, [number, number, number, number][]> = {
  rdylgn: [
    [  0, 165,   0,  38],
    [  5, 189,  24,  29],
    [ 10, 213,  48,  39],
    [ 15, 230,  82,  52],
    [ 20, 245, 115,  68],
    [ 25, 252, 152,  86],
    [ 30, 253, 185, 110],
    [ 35, 254, 212, 139],
    [ 40, 255, 235, 171],
    [ 45, 255, 251, 204],
    [ 50, 235, 248, 188],
    [ 55, 209, 238, 161],
    [ 60, 169, 220, 136],
    [ 65, 120, 198, 112],
    [ 70,  75, 176,  90],
    [ 75,  35, 152,  72],
    [ 80,   0, 125,  62],
    [ 85,   0, 104,  55],
    [ 90,   0,  81,  46],
    [100,   0,  68,  27],
  ],
  plasma: [
    [  0,  13,   8, 135],
    [ 10,  70,   4, 153],
    [ 20, 114,   1, 168],
    [ 30, 151,   5, 175],
    [ 40, 183,  29, 172],
    [ 50, 207,  54, 160],
    [ 60, 225,  78, 143],
    [ 70, 238, 103, 123],
    [ 80, 246, 128, 100],
    [ 90, 251, 153,  78],
    [100, 254, 177,  58],
  ],
  viridis: [
    [  0,  68,   1,  84],
    [ 10,  71,  22, 103],
    [ 20,  72,  40, 120],
    [ 30,  69,  55, 129],
    [ 40,  63,  71, 136],
    [ 50,  50, 100, 142],
    [ 60,  38, 128, 142],
    [ 70,  33, 143, 141],
    [ 80,  55, 184, 120],
    [ 90, 116, 208,  85],
    [100, 253, 231,  37],
  ],
};

export const CMAP_GRADIENTS: Record<Cmap, string> = {
  rdylgn:  'linear-gradient(to right, #a50026, #f46d43, #fee08b, #d9ef8b, #006837)',
  plasma:  'linear-gradient(to right, #0d0887, #cc4778, #f89540, #f0f921)',
  viridis: 'linear-gradient(to right, #440154, #31688e, #35b779, #fde725)',
};

/**
 * Build a MapLibre raster-color expression.
 *
 * The R channel (0–255) is provided by raster-color-mix [255,0,0,0].
 * We rescale: score_0_100 = pixel_val * 100 / 255
 * Pixels with R=0 are transparent (no data).
 * Pixels with score < cutoff are also transparent.
 */
export function buildColormapExpression(cmap: Cmap, cutoff: number): unknown[] {
  const stops = STOPS[cmap] ?? STOPS.rdylgn;
  // Build: ['interpolate', ['linear'], ['*', ['band', 1], 100/255], ...stops]
  // but first: pixels with band(1)==0 → transparent, and score < cutoff → transparent
  const scaledCutoff = cutoff * 255 / 100;

  const interpolate: unknown[] = [
    'interpolate', ['linear'],
    ['*', ['band', 1], 100 / 255],
  ];
  for (const [v, r, g, b] of stops) {
    interpolate.push(v, `rgba(${r},${g},${b},1)`);
  }

  return [
    'case',
    ['==', ['band', 1], 0], 'rgba(0,0,0,0)',            // no-data
    ['<', ['band', 1], scaledCutoff], 'rgba(0,0,0,0)',  // below cutoff
    interpolate,
  ];
}
