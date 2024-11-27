<template>
  <div class="debug-panel" v-if="isVisible">
    <div class="header">
      <h3>Debug Panel</h3>
      <button @click="togglePanel">{{ isExpanded ? 'Collapse' : 'Expand' }}</button>
    </div>

    <div v-if="isExpanded" class="content">
      <div class="section">
        <h4>Graph Stats</h4>
        <ul>
          <li>Nodes: {{ graphData?.nodes.length || 0 }}</li>
          <li>Edges: {{ graphData?.edges.length || 0 }}</li>
          <li>Metadata Keys: {{ Object.keys(graphData?.metadata || {}).length }}</li>
        </ul>
      </div>

      <div class="section">
        <h4>Binary Updates</h4>
        <ul>
          <li>Last Update: {{ formatTime(binaryStore.lastUpdateTime) }}</li>
          <li>Active Positions: {{ binaryStore.getAllPositions.length }}</li>
          <li>Initial Layout: {{ binaryStore.isInitial ? 'Yes' : 'No' }}</li>
          <li>Pending Updates: {{ binaryStore.pendingUpdateCount }}</li>
        </ul>
      </div>

      <div class="section">
        <h4>Sample Node</h4>
        <div v-if="sampleNode" class="sample-data">
          <p>ID: {{ sampleNode.id }}</p>
          <p>Position: {{ formatVector(sampleNode.position) }}</p>
          <p>Velocity: {{ formatVector(sampleNode.velocity) }}</p>
          <p>Edges: {{ sampleNode.edges.length }}</p>
        </div>
        <p v-else>No nodes available</p>
      </div>

      <div class="section">
        <h4>WebSocket Status</h4>
        <ul>
          <li>Connected: {{ wsStore.isConnected ? 'Yes' : 'No' }}</li>
          <li>Last Message: {{ formatTime(wsStore.lastMessageTime) }}</li>
          <li>Messages Sent: {{ wsStore.messageCount }}</li>
          <li>Queue Size: {{ wsStore.queueSize }}</li>
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
import { defineComponent, ref, computed } from 'vue'
import { useGraphDataManager } from '../services/graphDataManager'
import { useBinaryUpdateStore } from '../stores/binaryUpdate'
import { useWebSocketStore } from '../stores/websocket'

export default defineComponent({
  name: 'DebugPanel',

  setup() {
    const isVisible = ref(process.env.NODE_ENV === 'development')
    const isExpanded = ref(false)
    const graphManager = useGraphDataManager()
    const binaryStore = useBinaryUpdateStore()
    const wsStore = useWebSocketStore()

    const graphData = computed(() => graphManager.getGraphData())
    const sampleNode = computed(() => graphData.value?.nodes[0])

    const togglePanel = () => {
      isExpanded.value = !isExpanded.value
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

    const requestInitialData = () => {
      console.log('Requesting initial data...')
      graphManager.requestInitialData()
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
      isVisible,
      isExpanded,
      graphData,
      sampleNode,
      binaryStore,
      wsStore,
      togglePanel,
      formatTime,
      formatVector,
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
