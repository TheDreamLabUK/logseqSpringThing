import { defineConfig } from 'vite'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'client/iterate.html'),
      },
    },
  },
  server: {
    port: 3000,
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './client'),
    },
  },
})
