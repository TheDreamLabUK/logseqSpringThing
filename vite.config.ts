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
      outDir: resolve(__dirname, 'data/public/dist'),
      emptyOutDir: true,
      sourcemap: !isProd,
      // Disable all minification and optimization
      minify: false, 
      target: 'esnext', // Avoid transformations
      terserOptions: {
        compress: false,
        mangle: false
      },
      rollupOptions: {
        input: {
          main: resolve(__dirname, 'client/index.html')
        },
        output: {
          assetFileNames: (assetInfo) => {
            if (!assetInfo.name) return 'assets/[name][extname]';
            
            if (/\.(woff2?|eot|ttf|otf)$/i.test(assetInfo.name)) {
              return `assets/fonts/[name][extname]`;
            }
            if (/\.(css)$/i.test(assetInfo.name)) {
              return `assets/css/[name][extname]`;
            }
            if (/\.(png|jpe?g|gif|svg|ico)$/i.test(assetInfo.name)) {
              return `assets/images/[name][extname]`;
            }
            return `assets/[name][extname]`;
          },
          chunkFileNames: 'assets/js/[name]-[hash].js',
          entryFileNames: 'assets/js/[name]-[hash].js'
        }
      }
    },

    resolve: {
      alias: {
        '@': resolve(__dirname, './client')
      }
    },

    server: {
      port: 3001,
      host: true,
      proxy: {
        '/wss': {  // Updated from /ws to /wss to match nginx
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
      include: ['three'],
      disabled: true // Disable dependency optimization
    },

    define: {
      __QUEST__: isQuest,
      __DEV__: !isProd
    }
  };
});
