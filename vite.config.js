import { defineConfig } from 'vite';
import { createHtmlPlugin } from 'vite-plugin-html';

export default defineConfig({
  root: 'data/public',
  plugins: [createHtmlPlugin()],
  build: {
    outDir: './dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        globals: {
          'three': 'THREE'
        }
      }
    }
  },
  publicDir: 'assets',
  resolve: {
    alias: {
      '@': '/data/public',  // Add an alias for cleaner imports
    }
  },
  server: {
    open: true,
    port: 3000
  },
  optimizeDeps: {
    include: ['three'],  // Ensure Three.js is optimized
  }
});
