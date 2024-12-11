import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }) => {
  const isDev = mode === 'development'
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '')

  return {
    root: path.resolve(__dirname, 'client'),
    publicDir: path.resolve(__dirname, 'client/public'),
    plugins: [
      vue({
        script: {
          defineModel: true,
          propsDestructure: true
        }
      })
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './client'),
        '@components': path.resolve(__dirname, './client/components'),
        '@types': path.resolve(__dirname, './client/types'),
        '@stores': path.resolve(__dirname, './client/stores'),
        '@composables': path.resolve(__dirname, './client/composables'),
        '@platform': path.resolve(__dirname, './client/platform'),
        '@visualization': path.resolve(__dirname, './client/visualization'),
        'three/examples': path.resolve(__dirname, 'node_modules/three/examples'),
        'three': path.resolve(__dirname, 'node_modules/three'),
        'oimo': path.resolve(__dirname, 'node_modules/oimo/build/oimo.js'),
        'dat.gui': path.resolve(__dirname, 'node_modules/dat.gui/build/dat.gui.module.js')
      },
      extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json', '.vue']
    },
    build: {
      outDir: path.resolve(__dirname, 'dist'),
      emptyOutDir: true,
      assetsDir: 'assets',
      sourcemap: true,
      minify: mode === 'production' ? 'terser' : false,
      terserOptions: {
        compress: {
          drop_console: false,  // Keep console logs
          drop_debugger: false, // Keep debugger statements
          pure_funcs: []  // Don't remove any console functions
        },
        format: {
          comments: true  // Keep comments for better debugging
        },
        mangle: true  // Still mangle names for some optimization
      },
      rollupOptions: {
        input: {
          main: path.resolve(__dirname, 'client/index.html'),
          test: path.resolve(__dirname, 'client/indexTest.html'),
          iterate: path.resolve(__dirname, 'client/iterate.html')  // Add iterate entry point
        },
        output: {
          manualChunks: {
            'vendor-three': [
              'three',
              'three/examples/jsm/controls/OrbitControls',
              'three/examples/jsm/postprocessing/EffectComposer',
              'three/examples/jsm/postprocessing/RenderPass',
              'three/examples/jsm/postprocessing/UnrealBloomPass',
              'three/examples/jsm/postprocessing/ShaderPass',
              'three/examples/jsm/webxr/XRButton',
              'three/examples/jsm/webxr/XRControllerModelFactory',
              'three/examples/jsm/webxr/XRHandModelFactory'
            ],
            'vendor-vue': ['vue', 'pinia'],
            'vendor-utils': ['pako', 'oimo', 'dat.gui']
          },
          format: 'es',
          entryFileNames: 'assets/[name]-[hash].js',
          chunkFileNames: 'assets/[name]-[hash].js',
          assetFileNames: 'assets/[name]-[hash].[ext]'
        }
      }
    },
    optimizeDeps: {
      include: [
        'three',
        'three/examples/jsm/controls/OrbitControls',
        'three/examples/jsm/postprocessing/EffectComposer',
        'three/examples/jsm/postprocessing/RenderPass',
        'three/examples/jsm/postprocessing/UnrealBloomPass',
        'three/examples/jsm/postprocessing/SSAOPass',
        'three/examples/jsm/postprocessing/ShaderPass',
        'three/examples/jsm/shaders/FXAAShader',
        'three/examples/jsm/webxr/XRButton',
        'three/examples/jsm/webxr/XRControllerModelFactory',
        'three/examples/jsm/webxr/XRHandModelFactory',
        'oimo',
        'pako',
        'dat.gui'
      ],
      exclude: mode === 'quest' ? ['@oculus-native'] : []
    },
    server: {
      port: 4000,
      host: '0.0.0.0',
      fs: {
        strict: false
      },
      watch: {
        ignored: ['**/dist/**']
      }
    },
    define: {
      __VUE_OPTIONS_API__: false,
      __VUE_PROD_DEVTOOLS__: true,  // Enable Vue devtools in production
      __PLATFORM__: JSON.stringify(mode === 'quest' ? 'quest' : 'browser'),
      // Add global debug flag
      __DEBUG__: JSON.stringify(true)
    }
  }
})
