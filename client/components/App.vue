<template>
  <ErrorBoundary>
    <div id="app">
      <div id="scene-container" ref="sceneContainer">
        <canvas ref="canvas" />
        <GraphSystem 
          v-if="visualizationState.scene" 
          :visual-settings="visualSettings" 
        />
      </div>
      <ControlPanel />
      <div class="connection-status" :class="{ connected: isConnected }">
        WebSocket: {{ isConnected ? 'Connected' : 'Disconnected' }}
      </div>
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
      <div v-if="process.env.NODE_ENV === 'development'" class="debug-info">
        Scene Status: {{ visualizationState.isInitialized ? 'Ready' : 'Initializing' }}
        <br />
        Nodes: {{ visualizationStore.nodes.length }}
        <br />
        Edges: {{ visualizationStore.edges.length }}
      </div>
    </div>
  </ErrorBoundary>
</template>

<script lang="ts">
import { defineComponent, onMounted, onErrorCaptured, ref, onBeforeUnmount, ComponentPublicInstance, watch, computed, provide } from 'vue'
import { storeToRefs } from 'pinia'
import { useSettingsStore } from '../stores/settings'
import { useVisualizationStore } from '../stores/visualization'
import { useWebSocketStore } from '../stores/websocket'
import { useBinaryUpdateStore } from '../stores/binaryUpdate'
import ControlPanel from '@components/ControlPanel.vue'
import ErrorBoundary from '@components/ErrorBoundary.vue'
import GraphSystem from '@components/visualization/GraphSystem.vue'
import { errorTracking } from '../services/errorTracking'
import { useVisualization, SCENE_KEY } from '../composables/useVisualization'
import type { BaseMessage, GraphUpdateMessage, ErrorMessage, Node as WSNode, Edge as WSEdge, BinaryMessage } from '../types/websocket'
import type { Node as CoreNode, Edge as CoreEdge, GraphNode, GraphEdge, GraphData } from '../types/core'
import type { FisheyeConfig } from '../types/components'

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
  weight: wsNode.weight || 1,
  group: wsNode.group
});

// Transform websocket edge to core edge
const transformEdge = (wsEdge: WSEdge): CoreEdge => ({
  id: `${wsEdge.source}-${wsEdge.target}`,
  source: wsEdge.source,
  target: wsEdge.target,
  weight: wsEdge.weight || 1,
  width: wsEdge.width,
  color: wsEdge.color,
  type: wsEdge.type,
  metadata: wsEdge.metadata || {},
  userData: wsEdge.userData || {},
  directed: wsEdge.directed || false
});

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
    const canvas = ref<HTMLCanvasElement | null>(null)
    const error = ref<string | null>(null)

    // Get visualization settings
    const visualSettings = computed(() => {
      const settings = settingsStore.getVisualizationSettings;
      console.debug('Visualization settings:', {
        material: {
          metalness: settings.material.node_material_metalness,
          roughness: settings.material.node_material_roughness,
          opacity: settings.material.node_material_opacity
        },
        nodeColors: {
          base: settings.node_color,
          core: settings.node_color_core
        },
        sizes: {
          min: settings.min_node_size,
          max: settings.max_node_size
        }
      });
      return settings;
    });

    // Initialize visualization system
    const { initialize: initVisualization, updateNodes, updatePositions, state: visualizationState } = useVisualization()

    // Provide visualization state to child components
    provide('visualizationState', visualizationState)

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
              metadata: graphMsg.graphData.metadata ? Object.keys(graphMsg.graphData.metadata).length : 0,
              sampleNode: graphMsg.graphData.nodes?.[0] ? {
                id: graphMsg.graphData.nodes[0].id,
                position: graphMsg.graphData.nodes[0].position
              } : null
            })

            // Transform nodes and edges before setting graph data
            const transformedNodes = (graphMsg.graphData.nodes || []).map(transformNode)
            const transformedEdges = (graphMsg.graphData.edges || []).map(transformEdge)
            
            // Set graph data in store
            visualizationStore.setGraphData(
              transformedNodes,
              transformedEdges,
              graphMsg.graphData.metadata || {}
            )

            // Update visualization
            updateNodes(transformedNodes)

            // Log graph data state after update
            console.log('Graph data state after update:', {
              storeNodes: visualizationStore.nodes.length,
              storeEdges: visualizationStore.edges.length,
              graphData: visualizationStore.graphData ? {
                nodes: visualizationStore.graphData.nodes.length,
                edges: visualizationStore.graphData.edges.length
              } : null
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
          isInitial: data.isInitialLayout,
          sample: data.positions[0] ? {
            id: data.positions[0].id,
            position: [data.positions[0].x, data.positions[0].y, data.positions[0].z],
            velocity: [data.positions[0].vx, data.positions[0].vy, data.positions[0].vz]
          } : null
        })
        // Update visualization with position data
        updatePositions(data.positions, data.isInitialLayout)
        
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
        // Request initial data
        websocketStore.requestInitialData()
      })

      websocketStore.service.on('close', () => {
        console.log('WebSocket disconnected')
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
        if (canvas.value && sceneContainer.value) {
          console.log('Initializing visualization system...')
          
          // Set initial canvas size
          const rect = sceneContainer.value.getBoundingClientRect()
          canvas.value.width = rect.width
          canvas.value.height = rect.height
          
          await initVisualization({
            canvas: canvas.value,
            scene: {
              antialias: true,
              alpha: true,
              preserveDrawingBuffer: true,
              powerPreference: 'high-performance'
            }
          })
          console.log('Visualization system initialized')

          // Provide scene to child components
          if (visualizationState.value.scene) {
            provide(SCENE_KEY, visualizationState.value.scene)
          }
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
      canvas,
      isConnected,
      error,
      visualSettings,
      visualizationState,
      visualizationStore,
      process: {
        env: {
          NODE_ENV: process.env.NODE_ENV
        }
      }
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

#scene-container canvas {
  width: 100%;
  height: 100%;
  display: block;
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

.debug-info {
  position: fixed;
  top: 90px;
  right: 10px;
  padding: 10px;
  background-color: rgba(0, 0, 0, 0.8);
  color: white;
  border-radius: 4px;
  font-family: monospace;
  z-index: 1000;
  max-width: 300px;
  word-wrap: break-word;
}
</style>
