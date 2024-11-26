<template>
  <ErrorBoundary>
    <div id="app">
      <div id="scene-container" ref="sceneContainer"></div>
      <ControlPanel />
      <DebugPanel v-if="showDebugPanel" />
      <div v-if="showDebugPanel" class="connection-status" :class="{ connected: isConnected }">
        WebSocket: {{ isConnected ? 'Connected' : 'Disconnected' }}
      </div>
      <button 
        v-if="isQuest3WithPassthrough" 
        @click="toggleAR"
        class="ar-toggle-button"
        :class="{ active: isARActive }"
      >
        {{ isARActive ? 'Exit Mixed Reality' : 'Enter Mixed Reality' }}
      </button>
    </div>
  </ErrorBoundary>
</template>

<script lang="ts">
import { defineComponent, onMounted, onErrorCaptured, ref, computed, onBeforeUnmount, ComponentPublicInstance, watch } from 'vue'
import { useSettingsStore } from '../stores/settings'
import { useVisualizationStore } from '../stores/visualization'
import { useWebSocketStore } from '../stores/websocket'
import { usePlatform } from '../composables/usePlatform'
import ControlPanel from '@components/ControlPanel.vue'
import ErrorBoundary from '@components/ErrorBoundary.vue'
import DebugPanel from '@components/DebugPanel.vue'
import { errorTracking } from '../services/errorTracking'
import WebsocketService from '../services/websocketService'
import { useVisualization } from '../composables/useVisualization'
import { useForceGraph } from '../composables/useForceGraph'
import type { BaseMessage, GraphUpdateMessage, ErrorMessage, WebSocketEventMap, Node as WSNode, Edge as WSEdge } from '../types/websocket'
import type { Node as CoreNode, Edge as CoreEdge } from '../types/core'
import type { XRSession, XRSessionMode, QuestInitOptions } from '../types/platform/quest'

// Transform functions remain exactly the same
const transformNode = (wsNode: WSNode): CoreNode => ({
  id: wsNode.id,
  label: wsNode.label || wsNode.id,
  position: wsNode.position || [0, 0, 0],
  velocity: [0, 0, 0],
  size: wsNode.size || 1,
  color: wsNode.color,
  type: wsNode.type || 'default',
  metadata: wsNode.metadata || {},
  userData: wsNode.userData || {}
})

const transformEdge = (wsEdge: WSEdge): CoreEdge => ({
  id: `${wsEdge.source}-${wsEdge.target}`,
  source: wsEdge.source,
  target: wsEdge.target,
  weight: wsEdge.weight || 1,
  width: wsEdge.width || 1,
  color: wsEdge.color,
  type: wsEdge.type || 'default',
  metadata: wsEdge.metadata || {},
  userData: wsEdge.userData || {}
})

export default defineComponent({
  name: 'App',
  components: {
    ControlPanel,
    ErrorBoundary,
    DebugPanel
  },
  setup() {
    const settingsStore = useSettingsStore()
    const visualizationStore = useVisualizationStore()
    const websocketStore = useWebSocketStore()
    const { getPlatformInfo, hasXRSupport } = usePlatform()
    const websocketService = ref<WebsocketService | null>(null)
    const sceneContainer = ref<HTMLElement | null>(null)
    const forceGraph = ref<ReturnType<typeof useForceGraph> | null>(null)
    const isDebugMode = ref(
      window.location.search.includes('debug') || 
      process.env.NODE_ENV === 'development'
    )
    const isConnected = computed(() => websocketStore.connected)
    const isARActive = ref(false)

    // Enhanced Quest 3 detection with passthrough capability
    const isQuest3WithPassthrough = computed(() => {
      const ua = navigator.userAgent
      const isQuest3 = /Quest 3/i.test(ua)
      if (!isQuest3) return false

      // Check for color passthrough support
      return 'xr' in navigator && hasXRSupport() && (navigator as any).xr?.isSessionSupported('immersive-ar' as XRSessionMode)
    })

    // Initialize visualization system
    const { state: visualizationState, initialize: initVisualization } = useVisualization()

    // Show debug panel in development or when debug mode is enabled
    const showDebugPanel = computed(() => isDebugMode.value)

    // Toggle AR mode
    const toggleAR = async () => {
      try {
        if (!isARActive.value) {
          // Request AR session with passthrough
          const sessionInit: QuestInitOptions['xr'] = {
            optionalFeatures: ['local-floor', 'plane-detection', 'hand-tracking', 'layers', 'color-passthrough'],
            requiredFeatures: ['local-floor'],
            sessionMode: 'immersive-ar'
          }

          const session = await (navigator as any).xr?.requestSession('immersive-ar' as XRSessionMode, {
            ...sessionInit,
            domOverlay: { root: document.getElementById('app') }
          }) as XRSession

          if (session && visualizationState.value.renderer) {
            console.log('AR session started')
            isARActive.value = true
            
            // Set up session end handler
            session.addEventListener('end', () => {
              console.log('AR session ended')
              isARActive.value = false
            })

            // Bind session to renderer
            await visualizationState.value.renderer.xr.setSession(session)
          }
        } else {
          // End current AR session
          const session = visualizationState.value.renderer?.xr.getSession()
          if (session) {
            await session.end()
          }
        }
      } catch (error) {
        console.error('Error toggling AR mode:', error)
        errorTracking.trackError(error, {
          context: 'AR Toggle',
          component: 'App'
        })
      }
    }

    // Watch for graph data updates from the store
    watch(() => visualizationStore.nodes, (newNodes) => {
      if (forceGraph.value && newNodes.length > 0) {
        console.log('Updating force graph with nodes:', newNodes.length)
        forceGraph.value.updateGraph(newNodes, visualizationStore.edges)
      }
    }, { deep: true })

    onMounted(async () => {
      try {
        // Initialize settings with defaults
        settingsStore.applyServerSettings({})
        console.info('Settings initialized', {
          context: 'App Setup',
          settings: settingsStore.$state
        })

        // Initialize visualization system
        if (sceneContainer.value) {
          console.log('Initializing visualization system...')
          await initVisualization({
            canvas: document.createElement('canvas'),
            scene: {
              antialias: true,
              alpha: true,
              preserveDrawingBuffer: true,
              powerPreference: 'high-performance'
            }
          })

          // Initialize force graph after scene is ready
          if (visualizationState.value.scene) {
            console.log('Initializing force graph...')
            forceGraph.value = useForceGraph(visualizationState.value.scene)
          }
        }

        // Initialize WebSocket service
        console.log('Initializing WebSocket service...')
        websocketService.value = new WebsocketService()

        // Set up WebSocket event handlers
        websocketService.value.on('open', () => {
          console.log('WebSocket connected')
          websocketStore.setConnected(true)
          // Request initial data when connected
          websocketStore.requestInitialData()
        })

        websocketService.value.on('close', () => {
          console.log('WebSocket disconnected')
          websocketStore.setConnected(false)
        })

        websocketService.value.on('message', (data: BaseMessage) => {
          console.log('Received WebSocket message:', data)
          websocketStore.handleMessage(data)
        })

        websocketService.value.on('graphUpdate', ({ graphData }: WebSocketEventMap['graphUpdate']) => {
          console.log('Received graph update:', {
            nodes: graphData.nodes.length,
            edges: graphData.edges.length,
            metadata: graphData.metadata
          })
          visualizationStore.setGraphData(
            graphData.nodes.map(transformNode),
            graphData.edges.map(transformEdge),
            graphData.metadata
          )
        })

        websocketService.value.on('error', (error: ErrorMessage) => {
          console.error('WebSocket error:', error)
          websocketStore.setError(error.message || 'Unknown error')
          errorTracking.trackError(error, {
            context: 'WebSocket Error',
            component: 'App'
          })
        })

        // Log environment info
        console.info('Environment', {
          context: 'App Setup',
          nodeEnv: process.env.NODE_ENV,
          debug: isDebugMode.value,
          userAgent: navigator.userAgent,
          webGL: {
            renderer: document.createElement('canvas')
              .getContext('webgl2')
              ?.getParameter(WebGL2RenderingContext.RENDERER),
            vendor: document.createElement('canvas')
              .getContext('webgl2')
              ?.getParameter(WebGL2RenderingContext.VENDOR)
          }
        })

        // Set up window event listener for websocket messages
        window.addEventListener('websocket:send', ((event: CustomEvent) => {
          if (websocketService.value) {
            websocketService.value.send(event.detail)
          }
        }) as EventListener)

      } catch (error) {
        console.error('Error during App setup:', error)
        errorTracking.trackError(error, {
          context: 'App Setup',
          component: 'App'
        })
      }
    })

    onBeforeUnmount(() => {
      // Clean up WebSocket service
      if (websocketService.value) {
        websocketService.value.cleanup()
      }
      // Clean up force graph
      if (forceGraph.value) {
        forceGraph.value.dispose()
      }
    })

    // Additional error handling at app level
    onErrorCaptured((err, instance: ComponentPublicInstance | null, info) => {
      errorTracking.trackError(err, {
        context: 'App Root Error',
        component: (instance as any)?.$options?.name || 'Unknown',
        additional: { info }
      })
      // Let the error boundary handle it
      return false
    })

    // Add keyboard shortcut for toggling debug panel
    const handleKeyPress = (event: KeyboardEvent) => {
      // Ctrl/Cmd + Shift + D to toggle debug panel
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'D') {
        isDebugMode.value = !isDebugMode.value
        // Update URL to reflect debug mode
        const url = new URL(window.location.href)
        if (isDebugMode.value) {
          url.searchParams.set('debug', 'true')
        } else {
          url.searchParams.delete('debug')
        }
        window.history.replaceState({}, '', url.toString())
      }
    }

    onMounted(() => {
      window.addEventListener('keydown', handleKeyPress)
    })

    return {
      showDebugPanel,
      sceneContainer,
      isConnected,
      isQuest3WithPassthrough,
      isARActive,
      toggleAR
    }
  }
})
</script>

<style>
body, html {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow: hidden;
  background: #000000;
}

#scene-container {
  width: 100%;
  height: 100%;
  position: fixed;
  top: 0;
  left: 0;
  z-index: 0;
  background: #000000;
  touch-action: none;
}

#app {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
  pointer-events: none;
}

#app > * {
  pointer-events: auto;
}

.connection-status {
  position: fixed;
  top: 10px;
  right: 10px;
  padding: 5px 10px;
  background-color: rgba(0, 0, 0, 0.8);
  color: #ff4444;
  border-radius: 4px;
  font-family: monospace;
  z-index: 1000;
}

.connection-status.connected {
  color: #44ff44;
}

.ar-toggle-button {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  padding: 12px 24px;
  background-color: rgba(0, 0, 0, 0.8);
  color: #ffffff;
  border: 2px solid #ffffff;
  border-radius: 8px;
  font-size: 16px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
  z-index: 1000;
}

.ar-toggle-button:hover {
  background-color: rgba(255, 255, 255, 0.2);
}

.ar-toggle-button.active {
  background-color: rgba(255, 255, 255, 0.3);
  border-color: #44ff44;
}

/* Debug styles */
.debug-visible {
  border: 1px solid rgba(255, 0, 0, 0.3);
}

.debug-visible * {
  border: 1px solid rgba(0, 255, 0, 0.1);
}

/* Keyboard shortcut tooltip */
[data-tooltip] {
  position: relative;
}

[data-tooltip]:after {
  content: attr(data-tooltip);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  padding: 4px 8px;
  background-color: rgba(0, 0, 0, 0.8);
  color: white;
  font-size: 12px;
  border-radius: 4px;
  white-space: nowrap;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.2s, visibility 0.2s;
}

[data-tooltip]:hover:after {
  opacity: 1;
  visibility: visible;
}
</style>
