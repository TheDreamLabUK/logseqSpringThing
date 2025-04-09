
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: parseInt(process.env.VITE_DEV_SERVER_PORT || '3001'),
    strictPort: true,
    hmr: {
      port: 24678,
      protocol: 'ws',
    },
    proxy: {
      '/api': {
        target: `http://localhost:${process.env.VITE_API_PORT || '4000'}`,
        changeOrigin: true,
      },
      '/ws': {
        target: `ws://localhost:${process.env.VITE_API_PORT || '4000'}`,
        ws: true,
        changeOrigin: true,
      }
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
  },
});

