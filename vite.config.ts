import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }) => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '')

  return {
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
        'three/examples': path.resolve(__dirname, 'node_modules/three/examples'),
        'three': path.resolve(__dirname, 'node_modules/three')
      }
    },
    build: {
      target: 'esnext',
      minify: mode === 'production' ? 'terser' : false,
      sourcemap: mode === 'development',
      rollupOptions: {
        output: {
          manualChunks: {
            three: ['three'],
            'three-examples': [
              'three/examples/jsm/controls/OrbitControls',
              'three/examples/jsm/postprocessing/EffectComposer',
              'three/examples/jsm/postprocessing/RenderPass',
              'three/examples/jsm/postprocessing/UnrealBloomPass',
              'three/examples/jsm/postprocessing/SSAOPass',
              'three/examples/jsm/postprocessing/ShaderPass',
              'three/examples/jsm/shaders/FXAAShader'
            ],
            vue: ['vue', 'pinia'],
            platform: mode === 'quest' ? [
              'platform/quest',
              'xr/handTracking',
              'xr/spatialAudio'
            ] : [
              'platform/browser',
              'visualization/effects'
            ]
          },
          chunkSizeWarningLimit: 1000,
          dynamicImportVarsOptions: {
            warnOnError: true,
            exclude: []
          }
        }
      },
      terserOptions: {
        compress: {
          drop_console: mode === 'production',
          drop_debugger: mode === 'production',
          pure_funcs: mode === 'production' ? ['console.log', 'console.info'] : []
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
        'three/examples/jsm/shaders/FXAAShader'
      ],
      exclude: mode === 'quest' ? ['@oculus-native'] : []
    },
    server: {
      port: 3000,
      host: true,
      fs: {
        strict: false
      },
      watch: {
        ignored: ['**/dist/**']
      }
    },
    define: {
      __VUE_OPTIONS_API__: false,
      __VUE_PROD_DEVTOOLS__: mode !== 'production',
      __PLATFORM__: JSON.stringify(mode === 'quest' ? 'quest' : 'browser')
    }
  }
})
