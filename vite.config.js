// vite.config.js

import { defineConfig } from 'vite';
import three from 'vite-plugin-three';

export default defineConfig({
  root: 'data/public', // Set the root to the public directory
  build: {
    outDir: 'dist', // Output the build into a dist directory
    emptyOutDir: true, // Clear the output directory before building
    rollupOptions: {
      input: 'data/public/index.html', // Ensure correct entry point
    },
  },
  plugins: [three()], // Add the Three.js plugin
});