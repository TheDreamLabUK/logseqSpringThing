import { defineConfig } from 'vite';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig(({ mode, command }) => {
  const isProd = mode === 'production';
  const isQuest = process.env.npm_config_platform === 'quest';

  return {
    root: 'client',
    base: './',
    
    build: {
      outDir: '../dist',
      emptyOutDir: true,
      sourcemap: !isProd,
      minify: isProd ? 'terser' : false,
      terserOptions: {
        compress: {
          drop_console: isProd,
          drop_debugger: isProd
        }
      }
    },

    resolve: {
      alias: {
        '@': resolve(__dirname, './client')
      }
    },

    server: {
      port: 3000,
      host: true,
      proxy: {
        '/ws': {
          target: 'ws://localhost:4000',
          ws: true
        },
        '/api': {
          target: 'http://localhost:4000',
          changeOrigin: true
        }
      }
    },

    optimizeDeps: {
      include: ['three']
    },

    define: {
      __IS_QUEST__: isQuest,
      __DEV__: !isProd
    }
  };
});
