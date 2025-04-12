
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// Custom plugin to capture client-side console logs
const captureClientLogs = () => {
  return {
    name: 'capture-client-logs',
    transformIndexHtml(html) {
      return html.replace(
        '</head>',
        `<script>
          // Override console methods to ensure client logs are visible in the terminal
          const originalConsole = {
            log: console.log,
            info: console.info,
            warn: console.warn,
            error: console.error,
            debug: console.debug
          };

          // Send logs to the server
          function sendLogToServer(level, args) {
            try {
              const message = args.map(arg =>
                typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
              ).join(' ');

              fetch('/api/__log', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ level, message }),
              }).catch(() => {});
            } catch (e) {}
          }

          // Override console methods
          console.log = function(...args) {
            originalConsole.log.apply(console, args);
            sendLogToServer('log', args);
          };
          console.info = function(...args) {
            originalConsole.info.apply(console, args);
            sendLogToServer('info', args);
          };
          console.warn = function(...args) {
            originalConsole.warn.apply(console, args);
            sendLogToServer('warn', args);
          };
          console.error = function(...args) {
            originalConsole.error.apply(console, args);
            sendLogToServer('error', args);
          };
          console.debug = function(...args) {
            originalConsole.debug.apply(console, args);
            sendLogToServer('debug', args);
          };
        </script></head>`
      );
    },
    configureServer(server) {
      // Add endpoint to receive logs from client
      server.middlewares.use((req, res, next) => {
        if (req.url === '/api/__log' && req.method === 'POST') {
          let body = '';
          req.on('data', chunk => { body += chunk.toString(); });
          req.on('end', () => {
            try {
              const { level, message } = JSON.parse(body);
              console.log(`[CLIENT] [${level.toUpperCase()}] ${message}`);
              res.statusCode = 200;
              res.end('OK');
            } catch (e) {
              res.statusCode = 400;
              res.end('Bad Request');
            }
          });
        } else {
          next();
        }
      });
    }
  };
};

export default defineConfig({
  plugins: [react(), captureClientLogs()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
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
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('[PROXY ERROR]', err);
          });
        },
      },
      '/ws': {
        target: `ws://localhost:${process.env.VITE_API_PORT || '4000'}`,
        ws: true,
        changeOrigin: true,
      }
    },
    // Log all server events
    logger: {
      level: 'info',
      timestamp: true,
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});

