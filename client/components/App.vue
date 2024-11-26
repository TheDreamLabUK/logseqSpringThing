<template>
  <ErrorBoundary>
    <div id="app">
      <div id="scene-container" ref="sceneContainer"></div>
      <ControlPanel />
      <DebugPanel v-if="showDebugPanel" />
      <div class="connection-status" :class="{ connected: isConnected }">
        WebSocket: {{ isConnected ? 'Connected' : 'Disconnected' }}
      </div>
    </div>
  </ErrorBoundary>
</template>

<script lang="ts">
import { defineComponent, onMounted, onErrorCaptured, ref, computed, onBeforeUnmount, ComponentPublicInstance, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useSettingsStore } from '../stores/settings'
import { useVisualizationStore } from '../stores/visualization'
import { useWebSocketStore } from '../stores/websocket'
import ControlPanel from '@components/ControlPanel.vue'
import ErrorBoundary from '@components/ErrorBoundary.vue'
import DebugPanel from '@components/DebugPanel.vue'
import { errorTracking } from '../services/errorTracking'
import { useVisualization } from '../composables/useVisualization'
import type { BaseMessage, GraphUpdateMessage, ErrorMessage, Node as WSNode, Edge as WSEdge } from '../types/websocket'
import type { Node as CoreNode, Edge as CoreEdge } from '../types/core'

// Transform websocket node to core node
const transformNode = (wsNode: WSNode): CoreNode => ({
  id: wsNode.id,
  label: wsNode.label || wsNode.id,
  position: wsNode.position,
  velocity: wsNode.velocity,
  size: wsNode.size,
  color: wsNode.color,
  type: wsNode.type,
  metadata: wsNode.metadata || {},
  userData: wsNode.userData || {}
})

// Transform websocket edge to core edge
const transformEdge = (wsEdge: WSEdge): CoreEdge => ({
  id: `${wsEdge.source}-${wsEdge.target}`,
  source: wsEdge.source,
  target: wsEdge.target,
  weight: wsEdge.weight,
  width: wsEdge.width,
  color: wsEdge.color,
  type: wsEdge.type,
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
    // Initialize stores
    const settingsStore = useSettingsStore()
    const visualizationStore = useVisualizationStore()
    const websocketStore = useWebSocketStore()

    // Get reactive refs from stores
    const { connected: isConnected } = storeToRefs(websocketStore)
    
    const sceneContainer = ref<HTMLElement | null>(null)
    const isDebugMode = ref(
      window.location.search.includes('debug') || 
      process.env.NODE_ENV === 'development'
    )

    // Initialize visualization system
    const { initialize: initVisualization, updateNodes } = useVisualization()

    // Show debug panel in development or when debug mode is enabled
    const showDebugPanel = computed(() => isDebugMode.value)

    // Watch for graph data updates from the store
    watch(() => visualizationStore.nodes, (newNodes) => {
      if (newNodes.length > 0) {
        console.log('Updating visualization with nodes:', newNodes.length)
        updateNodes(newNodes)
      }
    }, { deep: true })

    // Set up graph update handler
    if (websocketStore.service) {
      websocketStore.service.on('graphUpdate', ({ graphData }: GraphUpdateMessage) => {
        if (!graphData) {
          console.warn('Received graph update with no data')
          return
        }

        console.log('Received graph update:', {
          nodes: graphData.nodes?.length || 0,
          edges: graphData.edges?.length || 0,
          metadata: graphData.metadata ? Object.keys(graphData.metadata).length : 0
        })

        // Transform nodes and edges before setting graph data
        const transformedNodes = (graphData.nodes || []).map(transformNode)
        const transformedEdges = (graphData.edges || []).map(transformEdge)
        visualizationStore.setGraphData(
          transformedNodes,
          transformedEdges,
          graphData.metadata || {}
        )
      })
    }

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
        }

        // Initialize WebSocket through store
        await websocketStore.initialize()

        // Log environment info
        console.info('Application initialized', {
          context: 'App Initialization',
          environment: process.env.NODE_ENV,
          debug: isDebugMode.value
        })

        // Set up window event listener for websocket messages
        window.addEventListener('websocket:send', ((event: CustomEvent) => {
          websocketStore.send(event.detail)
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
      // Clean up WebSocket through store
      websocketStore.cleanup()

      // Remove event listeners
      window.removeEventListener('websocket:send', ((event: CustomEvent) => {
        websocketStore.send(event.detail)
      }) as EventListener)
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

    onBeforeUnmount(() => {
      window.removeEventListener('keydown', handleKeyPress)
    })

    return {
      showDebugPanel,
      sceneContainer,
      isConnected
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
