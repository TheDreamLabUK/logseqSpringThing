import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { createHtmlPlugin } from 'vite-plugin-html';
import path from 'path';

export default defineConfig({
  root: path.resolve(__dirname, 'data/public'),
  plugins: [
    vue(),
    createHtmlPlugin({
      minify: {
        collapseWhitespace: true,
        removeComments: true,
        // Preserve JavaScript and CSS
        minifyJS: false,
        minifyCSS: true
      },
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
    sourcemap: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'data/public/index.html'),
      },
      output: {
        manualChunks: {
          'vendor-three': ['three'],
          'vendor-vue': ['vue'],
          'vendor-utils': ['pako']
        },
        format: 'es',
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]'
      }
    },
    chunkSizeWarningLimit: 1000,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: false,
        drop_debugger: true
      },
      format: {
        comments: false
      }
    }
  },
  base: '/',
  publicDir: path.resolve(__dirname, 'data/public/assets'),
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'data/public/js'),
      'vue': 'vue/dist/vue.runtime.esm-bundler.js'  // Use runtime build
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
