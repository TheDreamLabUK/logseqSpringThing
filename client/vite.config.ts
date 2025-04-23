
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    host: '0.0.0.0',
    // Use the VITE_DEV_SERVER_PORT from env (should be 5173 now)
    port: parseInt(process.env.VITE_DEV_SERVER_PORT || '5173'),
    strictPort: true,
    hmr: {
      // HMR port is internal (24678), client connects via Nginx (3001)
      port: parseInt(process.env.VITE_HMR_PORT || '24678'),
      // Let client infer host from window.location
      protocol: 'ws',
      clientPort: 3001, // Client connects to Nginx port
      path: '/ws' // Explicitly set the path Nginx proxies
    },
    // Proxy is now handled by Nginx, remove proxy config from Vite
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});

