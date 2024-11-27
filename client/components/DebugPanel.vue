<template>
  <div class="debug-panel" v-if="state.isVisible">
    <div class="header">
      <h3>Debug Panel</h3>
      <button @click="togglePanel">{{ state.isExpanded ? 'Collapse' : 'Expand' }}</button>
    </div>

    <div v-if="state.isExpanded" class="content">
      <div class="section">
        <h4>Graph Stats</h4>
        <ul>
          <li>Nodes: {{ state.metrics.nodeCount }}</li>
          <li>Edges: {{ state.metrics.edgeCount }}</li>
          <li>Metadata Keys: {{ Object.keys(graphData?.metadata || {}).length }}</li>
        </ul>
      </div>

      <div class="section">
        <h4>Binary Updates</h4>
        <ul>
          <li>Last Update: {{ formatTime(state.binaryStatus.lastUpdateTime) }}</li>
          <li>Active Positions: {{ state.binaryStatus.activePositions }}</li>
          <li>Initial Layout: {{ state.binaryStatus.isInitialLayout ? 'Yes' : 'No' }}</li>
          <li>Pending Updates: {{ state.binaryStatus.pendingCount }}</li>
        </ul>
      </div>

      <div class="section">
        <h4>Performance</h4>
        <ul>
          <li>Cache Hit Rate: {{ formatPercent(state.metrics.cacheHitRate) }}</li>
          <li>Update Interval: {{ formatTime(state.metrics.updateInterval) }}</li>
          <li>Position Updates: {{ state.metrics.positionUpdates }}</li>
          <li>FPS: {{ state.metrics.fps.toFixed(1) }}</li>
        </ul>
      </div>

      <div class="section">
        <h4>Sample Node</h4>
        <div v-if="state.sampleNode" class="sample-data">
          <p>ID: {{ state.sampleNode.id }}</p>
          <p>Position: {{ formatVector(state.sampleNode.position) }}</p>
          <p>Velocity: {{ formatVector(state.sampleNode.velocity) }}</p>
          <p>Edges: {{ state.sampleNode.edgeCount }}</p>
          <p v-if="state.sampleNode.weight">Weight: {{ state.sampleNode.weight.toFixed(2) }}</p>
          <p v-if="state.sampleNode.group">Group: {{ state.sampleNode.group }}</p>
        </div>
        <p v-else>No nodes available</p>
      </div>

      <div class="section">
        <h4>WebSocket Status</h4>
        <ul>
          <li>Connected: {{ state.wsStatus.connected ? 'Yes' : 'No' }}</li>
          <li>Last Message: {{ formatTime(state.wsStatus.lastMessageTime) }}</li>
          <li>Messages Sent: {{ state.wsStatus.messageCount }}</li>
          <li>Queue Size: {{ state.wsStatus.queueSize }}</li>
        </ul>
      </div>

      <div class="section">
        <h4>Actions</h4>
        <button @click="requestInitialData">Request Initial Data</button>
        <button @click="clearBinaryStore">Clear Binary Store</button>
        <button @click="reconnectWebSocket">Reconnect WebSocket</button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, reactive, computed, watch } from 'vue'
import { useVisualizationStore } from '../stores/visualization'
import { useBinaryUpdateStore } from '../stores/binaryUpdate'
import { useWebSocketStore } from '../stores/websocket'
import { useGraphSystem } from '../composables/useGraphSystem'
import type { DebugPanelState, SampleNodeData } from '../types/debug'

export default defineComponent({
  name: 'DebugPanel',

  setup() {
    const visualizationStore = useVisualizationStore()
    const binaryStore = useBinaryUpdateStore()
    const wsStore = useWebSocketStore()
    const graphSystem = useGraphSystem()

    const state = reactive<DebugPanelState>({
      isVisible: process.env.NODE_ENV === 'development',
      isExpanded: false,
      metrics: {
        cacheHitRate: 0,
        updateInterval: 0,
        fps: 0,
        nodeCount: 0,
        edgeCount: 0,
        positionUpdates: 0,
        messageCount: 0,
        queueSize: 0
      },
      sampleNode: null,
      wsStatus: {
        connected: false,
        lastMessageTime: 0,
        messageCount: 0,
        queueSize: 0,
        pendingUpdates: 0
      },
      binaryStatus: {
        lastUpdateTime: 0,
        activePositions: 0,
        isInitialLayout: false,
        pendingCount: 0
      }
    })

    const graphData = computed(() => visualizationStore.getGraphData)

    // Update metrics
    watch([graphData, () => graphSystem.metrics.value, () => wsStore.connected], () => {
      if (!graphData.value) return

      const total = graphSystem.metrics.value.cacheHits + graphSystem.metrics.value.cacheMisses
      const hitRate = total > 0 ? graphSystem.metrics.value.cacheHits / total : 0

      state.metrics = {
        cacheHitRate: hitRate,
        updateInterval: Date.now() - graphSystem.lastUpdateTime.value,
        fps: state.metrics.updateInterval > 0 ? 1000 / state.metrics.updateInterval : 0,
        nodeCount: graphData.value.nodes.length,
        edgeCount: graphData.value.edges.length,
        positionUpdates: graphSystem.metrics.value.positionUpdates,
        messageCount: wsStore.messageCount,
        queueSize: wsStore.queueSize
      }

      // Update sample node
      const firstNode = graphData.value.nodes[0]
      if (firstNode) {
        state.sampleNode = {
          id: firstNode.id,
          position: firstNode.position,
          velocity: firstNode.velocity,
          edgeCount: firstNode.edges.length,
          weight: firstNode.weight,
          group: firstNode.group
        }
      }

      // Update WebSocket status
      state.wsStatus = {
        connected: wsStore.connected,
        lastMessageTime: wsStore.lastMessageTime,
        messageCount: wsStore.messageCount,
        queueSize: wsStore.queueSize,
        pendingUpdates: wsStore.queueSize // Use queueSize as pendingUpdates
      }

      // Update binary status
      state.binaryStatus = {
        lastUpdateTime: binaryStore.lastUpdateTime,
        activePositions: binaryStore.getAllPositions.length,
        isInitialLayout: binaryStore.isInitial,
        pendingCount: binaryStore.pendingUpdateCount
      }
    })

    const togglePanel = () => {
      state.isExpanded = !state.isExpanded
    }

    const formatTime = (timestamp: number) => {
      if (!timestamp) return 'Never'
      const diff = Date.now() - timestamp
      if (diff < 1000) return 'Just now'
      return `${Math.round(diff / 1000)}s ago`
    }

    const formatVector = (vec?: number[] | null) => {
      if (!vec) return 'N/A'
      return `[${vec.map(v => v.toFixed(2)).join(', ')}]`
    }

    const formatPercent = (value: number) => {
      return `${(value * 100).toFixed(1)}%`
    }

    const requestInitialData = () => {
      console.log('Requesting initial data...')
      wsStore.requestInitialData()
    }

    const clearBinaryStore = () => {
      console.log('Clearing binary store...')
      binaryStore.clear()
    }

    const reconnectWebSocket = () => {
      console.log('Reconnecting WebSocket...')
      wsStore.reconnect()
    }

    return {
      state,
      graphData,
      togglePanel,
      formatTime,
      formatVector,
      formatPercent,
      requestInitialData,
      clearBinaryStore,
      reconnectWebSocket
    }
  }
})
</script>

<style scoped>
.debug-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: 300px;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 10px;
  font-family: monospace;
  z-index: 1000;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.header h3 {
  margin: 0;
}

.section {
  margin-bottom: 15px;
}

.section h4 {
  margin: 0 0 5px 0;
  color: #00ff00;
}

ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

li {
  margin: 2px 0;
  font-size: 12px;
}

.sample-data {
  font-size: 12px;
}

.sample-data p {
  margin: 2px 0;
}

button {
  background: #333;
  color: white;
  border: 1px solid #666;
  padding: 4px 8px;
  margin: 2px;
  cursor: pointer;
}

button:hover {
  background: #444;
}
</style>
