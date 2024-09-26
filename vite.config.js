import { defineConfig } from 'vite';
import { createHtmlPlugin } from 'vite-plugin-html';

export default defineConfig({
  root: 'data/public',  // Root directory for your frontend files, including index.html
  plugins: [createHtmlPlugin()],
  build: {
    outDir: '../../dist',  // Output to dist folder at the root level
    emptyOutDir: true,     // Clear the output directory before building
    // Remove rollupOptions.input, let Vite handle it automatically
  },
  publicDir: 'assets',  // Ensure static assets are in data/public/assets
});
