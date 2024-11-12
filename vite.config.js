import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { createHtmlPlugin } from 'vite-plugin-html';
import path from 'path';

export default defineConfig({
  root: path.resolve(__dirname, 'data/public'),
  plugins: [
    vue(),
    createHtmlPlugin({
      minify: true,
      inject: {
        data: {
          title: 'WebXR Graph Visualization'
        }
      }
    }),
  ],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    assetsDir: 'assets',
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'data/public/index.html'),
      },
      output: {
        manualChunks: (id) => {
          // Three.js and related modules
          if (id.includes('three') || id.includes('OrbitControls') || 
              id.includes('EffectComposer') || id.includes('RenderPass') || 
              id.includes('UnrealBloomPass')) {
            return 'vendor-three';
          }
          // Vue and related modules
          if (id.includes('vue') || id.includes('runtime-dom') || 
              id.includes('runtime-core')) {
            return 'vendor-vue';
          }
          // Other dependencies
          if (id.includes('pako')) {
            return 'vendor-utils';
          }
        },
        globals: {
          'three': 'THREE'
        }
      }
    },
    chunkSizeWarningLimit: 1000,
  },
  base: '/',
  publicDir: path.resolve(__dirname, 'data/public/assets'),
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'data/public/js'),
      'vue': 'vue/dist/vue.esm-bundler.js'
    },
    extensions: ['.js', '.json', '.vue']
  },
  server: {
    open: true,
    port: 3000
  },
  optimizeDeps: {
    include: ['three', 'vue', 'pako'],
    exclude: []
  }
});
