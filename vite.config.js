import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { createHtmlPlugin } from 'vite-plugin-html';
import path from 'path';

export default defineConfig({
  root: 'data/public',
  plugins: [
    vue(),
    createHtmlPlugin(),
  ],
  build: {
    outDir: '../dist', // Output to data/dist
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'data/public/index.html'),
      },
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
      '@': path.resolve(__dirname, 'data/public'),
      'vue': 'vue/dist/vue.esm-bundler.js'
    }
  },
  server: {
    open: true,
    port: 3000
  },
  optimizeDeps: {
    include: ['three', 'vue'],
  }
});