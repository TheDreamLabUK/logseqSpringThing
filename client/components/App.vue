<template>
  <ErrorBoundary>
    <div id="app">
      <div id="scene-container" ref="sceneContainer">
        <GraphSystem :visual-settings="visualSettings" />
      </div>
      <ControlPanel />
      <div class="connection-status" :class="{ connected: isConnected }">
        WebSocket: {{ isConnected ? 'Connected' : 'Disconnected' }}
      </div>
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
    </div>
  </ErrorBoundary>
</template>

<script lang="ts">
import { defineComponent, onMounted, onErrorCaptured, ref, onBeforeUnmount, ComponentPublicInstance, watch, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useSettingsStore } from '../stores/settings'
import { useVisualizationStore } from '../stores/visualization'
import { useWebSocketStore } from '../stores/websocket'
import { useBinaryUpdateStore } from '../stores/binaryUpdate'
import ControlPanel from '@components/ControlPanel.vue'
import ErrorBoundary from '@components/ErrorBoundary.vue'
import GraphSystem from '@components/visualization/GraphSystem.vue'
import { errorTracking } from '../services/errorTracking'
import { useVisualization } from '../composables/useVisualization'
import type { BaseMessage, GraphUpdateMessage, ErrorMessage, Node as WSNode, Edge as WSEdge, BinaryMessage, FisheyeUpdateMessage } from '../types/websocket'
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
  userData: wsNode.userData || {},
  weight: wsNode.weight,
  group: wsNode.group
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
  userData: wsEdge.userData || {},
  directed: wsEdge.directed
})

export default defineComponent({
  name: 'App',
  components: {
    ControlPanel,
    ErrorBoundary,
    GraphSystem
  },
  setup() {
    // Initialize stores
    const settingsStore = useSettingsStore()
    const visualizationStore = useVisualizationStore()
    const websocketStore = useWebSocketStore()
    const binaryUpdateStore = useBinaryUpdateStore()

    // Get reactive refs from stores
    const { connected: isConnected } = storeToRefs(websocketStore)
    
    const sceneContainer = ref<HTMLElement | null>(null)
    const error = ref<string | null>(null)

    // Get visualization settings
    const visualSettings = computed(() => settingsStore.getVisualizationSettings)

    // Initialize visualization system
    const { initialize: initVisualization, updateNodes, updatePositions } = useVisualization()

    // Watch for graph data updates from the store
    watch(() => visualizationStore.nodes, (newNodes) => {
      if (newNodes.length > 0) {
        console.log('Updating visualization with nodes:', newNodes.length)
        updateNodes(newNodes)
        // Clear transient position updates when receiving full mesh update
        binaryUpdateStore.clear()
      }
    }, { deep: true })

    // Watch for binary position updates
    watch(() => binaryUpdateStore.getAllPositions, (positions) => {
      if (positions.length > 0) {
        updatePositions(positions, binaryUpdateStore.isInitial)
      }
    })

    // Set up WebSocket message handlers
    if (websocketStore.service) {
      // Handle JSON messages
      websocketStore.service.on('message', (message: BaseMessage) => {
        console.debug('Received message:', message)
        switch (message.type) {
          case 'graphUpdate':
          case 'graphData':
            const graphMsg = message as GraphUpdateMessage
            if (!graphMsg.graphData) {
              console.warn('Received graph update with no data')
              return
            }

            console.log('Received graph update:', {
              nodes: graphMsg.graphData.nodes?.length || 0,
              edges: graphMsg.graphData.edges?.length || 0,
              metadata: graphMsg.graphData.metadata ? Object.keys(graphMsg.graphData.metadata).length : 0
            })

            // Transform nodes and edges before setting graph data
            const transformedNodes = (graphMsg.graphData.nodes || []).map(transformNode)
            const transformedEdges = (graphMsg.graphData.edges || []).map(transformEdge)
            visualizationStore.setGraphData(
              transformedNodes,
              transformedEdges,
              graphMsg.graphData.metadata || {}
            )
            break

          case 'fisheye_settings_updated':
            const fisheyeMsg = message as FisheyeUpdateMessage
            visualizationStore.updateFisheyeSettings({
              enabled: fisheyeMsg.fisheye_enabled,
              strength: fisheyeMsg.fisheye_strength,
              focus_x: fisheyeMsg.fisheye_focus_x,
              focus_y: fisheyeMsg.fisheye_focus_y,
              focus_z: fisheyeMsg.fisheye_focus_z,
              radius: fisheyeMsg.fisheye_radius
            })
            break

          case 'error':
            const errorMsg = message as ErrorMessage
            error.value = errorMsg.message
            errorTracking.trackError(new Error(errorMsg.message), {
              context: 'WebSocket Error',
              component: 'App'
            })
            break

          case 'settings_updated':
            settingsStore.applyServerSettings(message.settings)
            break

          case 'position_update_complete':
            console.debug('Position update completed:', message.status)
            break
        }
      })

      // Handle binary messages (GPU position updates)
      websocketStore.service.on('gpuPositions', (data: BinaryMessage) => {
        console.debug('Received GPU positions update:', {
          positions: data.positions.length,
          isInitial: data.isInitialLayout
        })
        // Store transient position updates
        binaryUpdateStore.updatePositions(
          data.positions,
          data.isInitialLayout
        )
      })

      // Handle connection events
      websocketStore.service.on('open', () => {
        console.log('WebSocket connected')
        error.value = null
        // Request initial data on connection
        websocketStore.requestInitialData()
      })

      websocketStore.service.on('close', () => {
        console.log('WebSocket disconnected')
        // Clear transient data on disconnect
        binaryUpdateStore.clear()
      })

      websocketStore.service.on('error', (err: ErrorMessage) => {
        console.error('WebSocket error:', err)
        error.value = err.message
        errorTracking.trackError(new Error(err.message), {
          context: 'WebSocket Error',
          component: 'App'
        })
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
          const canvas = document.createElement('canvas')
          sceneContainer.value.appendChild(canvas)
          await initVisualization({
            canvas,
            scene: {
              antialias: true,
              alpha: true,
              preserveDrawingBuffer: true,
              powerPreference: 'high-performance'
            }
          })
          console.log('Visualization system initialized')
        }

        // Initialize WebSocket through store
        await websocketStore.initialize()

        // Log environment info
        console.info('Application initialized', {
          context: 'App Initialization',
          environment: process.env.NODE_ENV
        })

      } catch (err) {
        console.error('Error during App setup:', err)
        error.value = err instanceof Error ? err.message : 'Unknown error during setup'
        errorTracking.trackError(err, {
          context: 'App Setup',
          component: 'App'
        })
      }
    })

    onBeforeUnmount(() => {
      // Clean up stores
      websocketStore.cleanup()
      binaryUpdateStore.clear()
    })

    // Additional error handling at app level
    onErrorCaptured((err, instance: ComponentPublicInstance | null, info) => {
      error.value = err instanceof Error ? err.message : 'An error occurred'
      errorTracking.trackError(err, {
        context: 'App Root Error',
        component: (instance as any)?.$options?.name || 'Unknown',
        additional: { info }
      })
      // Let the error boundary handle it
      return false
    })

    return {
      sceneContainer,
      isConnected,
      error,
      visualSettings
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

.error-message {
  position: fixed;
  top: 50px;
  right: 10px;
  padding: 10px;
  background-color: rgba(255, 0, 0, 0.8);
  color: white;
  border-radius: 4px;
  font-family: monospace;
  z-index: 1000;
  max-width: 300px;
  word-wrap: break-word;
}
</style>
