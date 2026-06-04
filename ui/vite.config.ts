import { defineConfig, createLogger } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import path from 'node:path';

const logger = createLogger();
const loggerError = logger.error.bind(logger);
logger.error = (msg, opts) => {
  if (opts?.error?.name === 'AbortError' || opts?.error?.message?.includes('cancelled')) return;
  loggerError(msg, opts);
};

export default defineConfig({
  customLogger: logger,
  plugins: [svelte()],
  root: 'src',
  build: {
    outDir: '../public',
    emptyOutDir: false,
    rollupOptions: {
      input: { index: path.resolve('src/index.html') },
      external: ['maplibre-gl'],
    },
  },
  server: {
    allowedHosts: ['z640'],
    proxy: {
      '/api': 'http://localhost:3000',
      '/tile': 'http://localhost:3000',
      '/wms': 'http://localhost:3000',
      '/s1-tile': 'http://localhost:3000',
      '/ranking-tile': 'http://localhost:3000',
      '/pmtiles': 'http://localhost:3000',
      '/sentinel2_tiles.geojson': 'http://localhost:3000',
      '/sentinel2_tile_labels.geojson': 'http://localhost:3000',
      '/sentinel2_chunks.geojson': 'http://localhost:3000',
    },
  },
});
