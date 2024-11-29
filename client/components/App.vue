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
import { SERVER_MESSAGE_TYPES, MESSAGE_FIELDS, ENABLE_BINARY_DEBUG } from '../constants/websocket'
import type { BaseMessage, GraphUpdateMessage, ErrorMessage, Node as WSNode, Edge as WSEdge, BinaryMessage } from '../types/websocket'
import type { Node as CoreNode, Edge as CoreEdge, GraphNode, GraphEdge, GraphData } from '../types/core'
import type { FisheyeConfig } from '../types/components'

// Transform functions unchanged
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
    // Store initializations unchanged
    const settingsStore = useSettingsStore()
    const visualizationStore = useVisualizationStore()
    const websocketStore = useWebSocketStore()
    const binaryUpdateStore = useBinaryUpdateStore()

    const { connected: isConnected } = storeToRefs(websocketStore)
    
    const sceneContainer = ref<HTMLElement | null>(null)
    const canvas = ref<HTMLCanvasElement | null>(null)
    const error = ref<string | null>(null)
    const isInitialDataRequested = ref(false)

    // Visualization settings unchanged
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

    const { initialize: initVisualization, updateNodes, updatePositions, state: visualizationState } = useVisualization()

    provide('visualizationState', visualizationState)

    const setupWebSocketHandlers = () => {
      if (!websocketStore.service) {
        console.error('WebSocket service not initialized');
        return;
      }

      // Connection event handlers
      websocketStore.service.on('open', () => {
        console.log('WebSocket connected');
        error.value = null;
        
        // Request initial data when connection is established
        if (!isInitialDataRequested.value) {
          console.debug('Requesting initial graph data');
          websocketStore.service?.send({
            type: SERVER_MESSAGE_TYPES.INITIAL_DATA
          });
          isInitialDataRequested.value = true;
        }
      });

      websocketStore.service.on('close', () => {
        console.log('WebSocket disconnected');
        binaryUpdateStore.clear();
        isInitialDataRequested.value = false;
      });

      websocketStore.service.on('error', (err: ErrorMessage) => {
        console.error('WebSocket error:', err);
        error.value = err.message;
        errorTracking.trackError(new Error(err.message), {
          context: 'WebSocket Error',
          component: 'App'
        });
      });

      // JSON message handler
      websocketStore.service.on('message', (message: BaseMessage) => {
        console.debug('Received message:', message);
        switch (message.type) {
          case SERVER_MESSAGE_TYPES.GRAPH_UPDATE:
            const graphMsg = message as GraphUpdateMessage;
            const graphData = graphMsg.graphData || graphMsg[MESSAGE_FIELDS.GRAPH_DATA];
            if (!graphData) {
              console.warn('Received graph update with no data');
              return;
            }

            console.log('Received graph update:', {
              nodes: graphData.nodes?.length || 0,
              edges: graphData.edges?.length || 0,
              metadata: graphData.metadata ? Object.keys(graphData.metadata).length : 0,
              sampleNode: graphData.nodes?.[0] ? {
                id: graphData.nodes[0].id,
                position: graphData.nodes[0].position
              } : null
            });

            const transformedNodes = (graphData.nodes || []).map(transformNode);
            const transformedEdges = (graphData.edges || []).map(transformEdge);
            
            visualizationStore.setGraphData(
              transformedNodes,
              transformedEdges,
              graphData.metadata || {}
            );

            updateNodes(transformedNodes);

            console.log('Graph data state after update:', {
              storeNodes: visualizationStore.nodes.length,
              storeEdges: visualizationStore.edges.length,
              graphData: visualizationStore.graphData ? {
                nodes: visualizationStore.graphData.nodes.length,
                edges: visualizationStore.graphData.edges.length
              } : null
            });
            break;

          case SERVER_MESSAGE_TYPES.SETTINGS_UPDATED:
            settingsStore.applyServerSettings(message.settings);
            break;

          case SERVER_MESSAGE_TYPES.POSITION_UPDATE_COMPLETE:
            console.debug('Position update completed:', message[MESSAGE_FIELDS.STATUS]);
            break;
        }
      });

      // Binary message handler
      websocketStore.service.on('gpuPositions', (data: BinaryMessage) => {
        if (ENABLE_BINARY_DEBUG) {
          console.debug('Received GPU positions update:', {
            bufferSize: data.data.byteLength,
            nodeCount: visualizationStore.nodes.length
          });
        }

        // Update binary store with raw ArrayBuffer data
        binaryUpdateStore.updateFromBinary(data);

        // Get the processed TypedArrays from the store
        const positions = binaryUpdateStore.getAllPositions;
        const velocities = binaryUpdateStore.getAllVelocities;
        const nodeCount = visualizationStore.nodes.length;

        // Update visualization with TypedArrays and node count
        updatePositions(positions, velocities, nodeCount);
      });
    };

    // Rest of the component unchanged
    onMounted(async () => {
      try {
        settingsStore.applyServerSettings({});
        console.info('Settings initialized', {
          context: 'App Setup',
          settings: settingsStore.$state
        });

        if (canvas.value && sceneContainer.value) {
          console.log('Initializing visualization system...');
          
          const rect = sceneContainer.value.getBoundingClientRect();
          canvas.value.width = rect.width;
          canvas.value.height = rect.height;
          
          await initVisualization({
            canvas: canvas.value,
            scene: {
              antialias: true,
              alpha: true,
              preserveDrawingBuffer: true,
              powerPreference: 'high-performance'
            }
          });
          console.log('Visualization system initialized');

          if (visualizationState.value.scene) {
            provide(SCENE_KEY, visualizationState.value.scene);
          }
        }

        await websocketStore.initialize();
        setupWebSocketHandlers();

        console.info('Application initialized', {
          context: 'App Initialization',
          environment: process.env.NODE_ENV
        });

      } catch (err) {
        console.error('Error during App setup:', err);
        error.value = err instanceof Error ? err.message : 'Unknown error during setup';
        errorTracking.trackError(err, {
          context: 'App Setup',
          component: 'App'
        });
      }
    });

    onBeforeUnmount(() => {
      websocketStore.cleanup();
      binaryUpdateStore.clear();
    });

    onErrorCaptured((err, instance: ComponentPublicInstance | null, info) => {
      error.value = err instanceof Error ? err.message : 'An error occurred';
      errorTracking.trackError(err, {
        context: 'App Root Error',
        component: (instance as any)?.$options?.name || 'Unknown',
        additional: { info }
      });
      return false;
    });

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
    };
  }
});
</script>

<style>
/* Styles unchanged */
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
