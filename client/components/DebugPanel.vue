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
          <p>Position: {{ formatPosition(sampleNode) }}</p>
          <p>Velocity: {{ formatVelocity(sampleNode) }}</p>
        </div>
        <p v-else>No nodes available</p>
      </div>

      <div class="section">
        <h4>WebSocket Status</h4>
        <ul>
          <li>Connected: {{ wsStore.connected ? 'Yes' : 'No' }}</li>
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
import { defineComponent, ref, computed, onMounted } from 'vue'
import { useBinaryUpdateStore } from '../stores/binaryUpdate'
import { useWebSocketStore } from '../stores/websocket'
import type { PositionUpdate } from '../types/websocket'

export default defineComponent({
  name: 'DebugPanel',

  setup() {
    const isVisible = ref(process.env.NODE_ENV === 'development')
    const isExpanded = ref(false)
    const wsStore = useWebSocketStore()
    const binaryStore = useBinaryUpdateStore()
    
    // For now, we'll just show basic graph data from the binary store
    const graphData = computed(() => ({
      nodes: binaryStore.getAllPositions,
      edges: [],
      metadata: {}
    }))
    const sampleNode = computed(() => graphData.value.nodes[0])

    const togglePanel = () => {
      isExpanded.value = !isExpanded.value
    }

    const formatTime = (timestamp: number) => {
      if (!timestamp) return 'Never'
      const diff = Date.now() - timestamp
      if (diff < 1000) return 'Just now'
      return `${Math.round(diff / 1000)}s ago`
    }

    const formatPosition = (node: PositionUpdate) => {
      return `[${node.x.toFixed(2)}, ${node.y.toFixed(2)}, ${node.z.toFixed(2)}]`
    }

    const formatVelocity = (node: PositionUpdate) => {
      return `[${node.vx.toFixed(2)}, ${node.vy.toFixed(2)}, ${node.vz.toFixed(2)}]`
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

    onMounted(() => {
      if (!wsStore.connected) {
        wsStore.initialize()
      }
    })

    return {
      isVisible,
      isExpanded,
      graphData,
      sampleNode,
      binaryStore,
      wsStore,
      togglePanel,
      formatTime,
      formatPosition,
      formatVelocity,
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
