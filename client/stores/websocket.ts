import { defineStore } from 'pinia'
import WebsocketService from '../services/websocketService'
import { useVisualizationStore } from './visualization'
import { useBinaryUpdateStore } from './binaryUpdate'
import type { BaseMessage, ErrorMessage, GraphUpdateMessage, BinaryMessage, Edge as WsEdge } from '../types/websocket'
import type { Node, Edge } from '../types/core'

interface WebSocketState {
  connected: boolean
  error: string | null
  service: WebsocketService | null
  lastMessageTime: number
  messageCount: number
  queueSize: number
}

export const useWebSocketStore = defineStore('websocket', {
  state: (): WebSocketState => ({
    connected: false,
    error: null,
    service: null,
    lastMessageTime: 0,
    messageCount: 0,
    queueSize: 0
  }),

  actions: {
    async initialize() {
      if (this.service) {
        console.log('WebSocket service already initialized')
        return
      }

      const visualizationStore = useVisualizationStore()
      const binaryUpdateStore = useBinaryUpdateStore()

      this.service = new WebsocketService()
      
      // Set up event handlers
      this.service.on('open', () => {
        console.debug('WebSocket connected, requesting initial data')
        this.connected = true
        this.error = null
        // Request initial data immediately after connection
        this.requestInitialData()
      })

      this.service.on('close', () => {
        this.connected = false
      })

      this.service.on('error', (error: ErrorMessage) => {
        this.error = error.message || 'Unknown error'
      })

      // Handle graph updates
      this.service.on('graphUpdate', (message: GraphUpdateMessage) => {
        console.debug('Received graph update:', {
          nodeCount: (message.graphData || message.graph_data)?.nodes?.length || 0,
          edgeCount: (message.graphData || message.graph_data)?.edges?.length || 0,
          timestamp: new Date().toISOString()
        })

        // Get graph data from either camelCase or snake_case field
        const graphData = message.graphData || message.graph_data
        if (!graphData) {
          console.warn('No graph data found in message')
          return
        }
        
        // Convert websocket edges to core edges by adding IDs
        const edges: Edge[] = graphData.edges.map((edge: WsEdge) => ({
          ...edge,
          id: `${edge.source}-${edge.target}` // Generate ID from source and target
        }))

        // Update visualization store with new graph data
        visualizationStore.setGraphData(
          graphData.nodes as Node[],
          edges,
          graphData.metadata
        )
      })

      // Handle binary position updates
      this.service.on('gpuPositions', (message: BinaryMessage) => {
        console.debug('Received binary position update:', {
          positionCount: message.positions.length,
          isInitialLayout: message.isInitialLayout,
          timestamp: new Date().toISOString()
        })
        
        // Update binary update store
        binaryUpdateStore.updatePositions(message.positions, message.isInitialLayout)
      })

      // Connect to server
      try {
        await this.service.connect()
      } catch (error) {
        this.error = error instanceof Error ? error.message : 'Unknown error'
        throw error
      }
    },

    setConnected(value: boolean) {
      this.connected = value
    },

    setError(message: string) {
      this.error = message
    },

    handleMessage(message: BaseMessage) {
      // Handle incoming messages
      if (message.type === 'error') {
        this.error = (message as ErrorMessage).message
      }
    },

    send(data: any) {
      if (!this.service) {
        console.error('Cannot send message: WebSocket service not initialized')
        return
      }
      this.service.send(data)
    },

    requestInitialData() {
      console.debug('Requesting initial graph data')
      this.send({ type: 'initial_data' })
    },

    async reconnect() {
      if (this.service) {
        this.service.cleanup()
        this.service = null
      }
      await this.initialize()
    },

    cleanup() {
      if (this.service) {
        this.service.cleanup()
        this.service = null
      }
      this.connected = false
      this.error = null
      this.lastMessageTime = 0
      this.messageCount = 0
      this.queueSize = 0
    }
  }
})
