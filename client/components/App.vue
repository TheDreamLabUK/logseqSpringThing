<template>
  <ErrorBoundary>
    <div id="app">
      <div id="scene-container" ref="sceneContainer">
        <canvas ref="canvas" />
        <GraphSystem 
          v-if="visualizationState.scene && isConnected" 
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
import type { BaseMessage, GraphUpdateMessage, ErrorMessage, Node, Edge, BinaryMessage } from '../types/websocket'
import type { GraphNode, GraphEdge, GraphData } from '../types/core'
import type { FisheyeConfig } from '../types/components'

export default defineComponent({
  name: 'App',
  components: {
    ControlPanel,
    ErrorBoundary,
    GraphSystem
  },
  setup() {
    const settingsStore = useSettingsStore()
    const visualizationStore = useVisualizationStore()
    const websocketStore = useWebSocketStore()
    const binaryUpdateStore = useBinaryUpdateStore()

    const { connected: isConnected } = storeToRefs(websocketStore)
    
    const sceneContainer = ref<HTMLElement | null>(null)
    const canvas = ref<HTMLCanvasElement | null>(null)
    const error = ref<string | null>(null)
    const isInitialDataRequested = ref(false)

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

      websocketStore.service.on('open', () => {
        console.log('WebSocket connected');
        error.value = null;
        
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

      websocketStore.service.on('message', (message: BaseMessage) => {
        console.debug('Received message:', message);
        switch (message.type) {
          case SERVER_MESSAGE_TYPES.GRAPH_UPDATE:
            const graphMsg = message as GraphUpdateMessage;
            if (!graphMsg.graphData) {
              console.warn('Received graph update with no data');
              return;
            }

            console.log('Received graph update:', {
              nodes: graphMsg.graphData.nodes?.length || 0,
              edges: graphMsg.graphData.edges?.length || 0,
              metadata: graphMsg.graphData.metadata ? Object.keys(graphMsg.graphData.metadata).length : 0,
              sampleNode: graphMsg.graphData.nodes?.[0] ? {
                id: graphMsg.graphData.nodes[0].id,
                position: graphMsg.graphData.nodes[0].position
              } : null
            });
            
            visualizationStore.setGraphData(
              graphMsg.graphData.nodes,
              graphMsg.graphData.edges,
              graphMsg.graphData.metadata || {}
            );

            updateNodes(graphMsg.graphData.nodes);

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

      websocketStore.service.on('gpuPositions', (data: BinaryMessage) => {
        if (ENABLE_BINARY_DEBUG) {
          console.debug('Received GPU positions update:', {
            bufferSize: data.data.byteLength,
            nodeCount: visualizationStore.nodes.length
          });
        }

        binaryUpdateStore.updateFromBinary(data);

        const positions = binaryUpdateStore.getAllPositions;
        const velocities = binaryUpdateStore.getAllVelocities;
        const nodeCount = visualizationStore.nodes.length;

        updatePositions(positions, velocities, nodeCount);
      });
    };

    const initializeApp = async () => {
      try {
        // First initialize websocket
        console.log('Initializing WebSocket connection...');
        await websocketStore.initialize();
        setupWebSocketHandlers();

        // Then initialize settings
        settingsStore.applyServerSettings({});
        console.info('Settings initialized', {
          context: 'App Setup',
          settings: settingsStore.$state
        });

        // Finally initialize visualization
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
    };

    onMounted(() => {
      initializeApp();
    });

    // Watch for websocket disconnection and try to reconnect
    watch(() => isConnected.value, (connected) => {
      if (!connected && !error.value) {
        console.log('WebSocket disconnected, attempting to reconnect...');
        websocketStore.reconnect();
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
