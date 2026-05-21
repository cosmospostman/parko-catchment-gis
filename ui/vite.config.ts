import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import path from 'node:path';

export default defineConfig({
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
    proxy: {
      '/api': 'http://localhost:3000',
      '/tile': 'http://localhost:3000',
      '/wms': 'http://localhost:3000',
      '/s1-tile': 'http://localhost:3000',
      '/ranking-tile': 'http://localhost:3000',
      '/sentinel2_tiles.geojson': 'http://localhost:3000',
      '/sentinel2_tile_labels.geojson': 'http://localhost:3000',
    },
  },
});
