import { defineStore } from 'pinia'
import WebsocketService from '../services/websocketService'
import { useVisualizationStore } from './visualization'
import { useBinaryUpdateStore } from './binaryUpdate'
import type { BaseMessage, ErrorMessage, GraphUpdateMessage, BinaryMessage, Edge as WsEdge, SimulationModeMessage } from '../types/websocket'
import type { Node, Edge } from '../types/core'

interface WebSocketState {
  connected: boolean
  error: string | null
  service: WebsocketService | null
  lastMessageTime: number
  messageCount: number
  queueSize: number
  connectionAttempts: number
  lastReconnectTime: number
  gpuEnabled: boolean
  initialDataRequested: boolean
  performanceMetrics: {
    avgMessageProcessingTime: number
    messageProcessingSamples: number[]
    avgPositionUpdateTime: number
    positionUpdateSamples: number[]
    lastPerformanceReset: number
  }
}

const MAX_PERFORMANCE_SAMPLES = 100;
const PERFORMANCE_RESET_INTERVAL = 60000; // Reset metrics every minute

export const useWebSocketStore = defineStore('websocket', {
  state: (): WebSocketState => ({
    connected: false,
    error: null,
    service: null,
    lastMessageTime: 0,
    messageCount: 0,
    queueSize: 0,
    connectionAttempts: 0,
    lastReconnectTime: 0,
    gpuEnabled: false,
    initialDataRequested: false,
    performanceMetrics: {
      avgMessageProcessingTime: 0,
      messageProcessingSamples: [],
      avgPositionUpdateTime: 0,
      positionUpdateSamples: [],
      lastPerformanceReset: Date.now()
    }
  }),

  getters: {
    isConnected: (state) => state.connected,
    hasError: (state) => state.error !== null,
    connectionHealth: (state) => {
      if (!state.connected) return 'disconnected'
      if (state.error) return 'error'
      if (state.connectionAttempts > 0) return 'unstable'
      return 'healthy'
    },
    performanceStatus: (state) => {
      const { avgMessageProcessingTime, avgPositionUpdateTime } = state.performanceMetrics
      if (avgMessageProcessingTime > 100 || avgPositionUpdateTime > 16) return 'poor'
      if (avgMessageProcessingTime > 50 || avgPositionUpdateTime > 8) return 'fair'
      return 'good'
    },
    isGPUEnabled: (state) => state.gpuEnabled
  },

  actions: {
    async initialize() {
      console.debug('[WebSocketStore] Initializing websocket store')
      if (this.service) {
        console.log('[WebSocketStore] WebSocket service already initialized')
        return
      }

      const visualizationStore = useVisualizationStore()
      const binaryUpdateStore = useBinaryUpdateStore()

      this.service = new WebsocketService()
      
      this._setupEventHandlers(visualizationStore, binaryUpdateStore)
      
      try {
        console.debug('[WebSocketStore] Attempting to connect websocket')
        await this.service.connect()
      } catch (error) {
        this._handleConnectionError(error)
        throw error
      }
    },

    _setupEventHandlers(visualizationStore: any, binaryUpdateStore: any) {
      if (!this.service) return

      this.service.on('open', () => {
        console.debug('[WebSocketStore] WebSocket connected, connection state:', {
          connected: this.connected,
          initialDataRequested: this.initialDataRequested,
          connectionAttempts: this.connectionAttempts
        })
        this.connected = true
        this.error = null
        this.connectionAttempts = 0
        if (!this.initialDataRequested) {
          this.requestInitialData()
        }
      })

      this.service.on('close', () => {
        console.debug('[WebSocketStore] WebSocket closed')
        this.connected = false
        this._handleDisconnect()
      })

      this.service.on('error', (error: ErrorMessage) => {
        this._handleError(error)
      })

      this.service.on('maxReconnectAttemptsReached', () => {
        this._handleMaxReconnectAttempts()
      })

      this.service.on('graphUpdate', (message: GraphUpdateMessage) => {
        const startTime = performance.now()
        console.debug('[WebSocketStore] Received graph update message:', {
          type: message.type,
          hasGraphData: !!message.graphData,
          hasSnakeCaseData: !!message.graph_data,
          timestamp: new Date().toISOString()
        })
        
        try {
          this._handleGraphUpdate(message, visualizationStore)
        } catch (error) {
          console.error('[WebSocketStore] Error processing graph update:', error)
        }

        this._updateMessageProcessingMetrics(performance.now() - startTime)
      })

      this.service.on('gpuPositions', (message: BinaryMessage) => {
        const startTime = performance.now()
        
        try {
          console.debug('[WebSocketStore] Processing GPU positions update:', {
            dataSize: message.data.byteLength,
            nodeCount: message.nodeCount,
            timestamp: new Date().toISOString()
          })
          binaryUpdateStore.updateFromBinary(message)
        } catch (error) {
          console.error('[WebSocketStore] Error processing position update:', error)
        }

        this._updatePositionUpdateMetrics(performance.now() - startTime)
      })

      // Handle simulation mode changes
      this.service.on('simulationModeSet', (mode: string) => {
        console.debug('[WebSocketStore] Setting simulation mode:', mode)
        visualizationStore.setSimulationMode(mode)
      })

      // Handle all messages to catch GPU state updates
      this.service.on('message', (message: BaseMessage) => {
        console.debug('[WebSocketStore] Received message:', {
          type: message.type,
          timestamp: new Date().toISOString()
        })
        
        if (message.type === 'gpu_state' && 'enabled' in message) {
          this.gpuEnabled = message.enabled
          console.debug(`[WebSocketStore] GPU acceleration ${message.enabled ? 'enabled' : 'disabled'}`)
        }
      })
    },

    _handleGraphUpdate(message: GraphUpdateMessage, visualizationStore: any) {
      console.debug('[WebSocketStore] Processing graph update:', {
        nodeCount: message.graphData?.nodes?.length || 0,
        edgeCount: message.graphData?.edges?.length || 0,
        hasMetadata: !!(message.graphData?.metadata || message.graph_data?.metadata),
        timestamp: new Date().toISOString()
      })

      if (!message.graphData && !message.graph_data) {
        console.warn('[WebSocketStore] No graph data found in message')
        return
      }

      const graphData = message.graphData || message.graph_data
      if (!graphData) return

      console.debug('[WebSocketStore] Graph data details:', {
        nodes: graphData.nodes?.map(n => ({ id: n.id, hasPosition: !!n.position })) || [],
        edges: graphData.edges?.map(e => ({ source: e.source, target: e.target })) || [],
        metadata: graphData.metadata
      })

      const edges: Edge[] = graphData.edges.map((edge: WsEdge) => ({
        ...edge,
        id: `${edge.source}-${edge.target}`
      }))

      visualizationStore.setGraphData(
        graphData.nodes as Node[],
        edges,
        graphData.metadata
      )
    },

    _handleConnectionError(error: unknown) {
      console.error('[WebSocketStore] Connection error:', error)
      this.error = error instanceof Error ? error.message : 'Unknown connection error'
      this.connectionAttempts++
      this.lastReconnectTime = Date.now()
    },

    _handleError(error: ErrorMessage) {
      this.error = error.message || 'Unknown error'
      if (error.code) {
        console.error(`[WebSocketStore] WebSocket error [${error.code}]:`, this.error)
      } else {
        console.error('[WebSocketStore] WebSocket error:', this.error)
      }
      if (error.details) {
        console.debug('[WebSocketStore] Error details:', error.details)
      }
    },

    _handleDisconnect() {
      const timeSinceLastReconnect = Date.now() - this.lastReconnectTime
      
      if (timeSinceLastReconnect > 60000) { // Reset counter if more than 1 minute since last reconnect
        this.connectionAttempts = 0
      }
      
      this.connectionAttempts++
      console.debug('[WebSocketStore] Handling disconnect:', {
        connectionAttempts: this.connectionAttempts,
        timeSinceLastReconnect,
        timestamp: new Date().toISOString()
      })
    },

    _handleMaxReconnectAttempts() {
      this.error = 'Maximum reconnection attempts reached. Please refresh the page.'
      console.error('[WebSocketStore] Max reconnection attempts reached')
    },

    _updateMessageProcessingMetrics(processingTime: number) {
      this._updateMetrics('message', processingTime)
    },

    _updatePositionUpdateMetrics(processingTime: number) {
      this._updateMetrics('position', processingTime)
    },

    _updateMetrics(type: 'message' | 'position', processingTime: number) {
      const metrics = this.performanceMetrics
      const now = Date.now()

      // Reset metrics if needed
      if (now - metrics.lastPerformanceReset > PERFORMANCE_RESET_INTERVAL) {
        metrics.messageProcessingSamples = []
        metrics.positionUpdateSamples = []
        metrics.lastPerformanceReset = now
      }

      if (type === 'message') {
        metrics.messageProcessingSamples.push(processingTime)
        if (metrics.messageProcessingSamples.length > MAX_PERFORMANCE_SAMPLES) {
          metrics.messageProcessingSamples.shift()
        }
        metrics.avgMessageProcessingTime = 
          metrics.messageProcessingSamples.reduce((a, b) => a + b, 0) / 
          metrics.messageProcessingSamples.length
      } else {
        metrics.positionUpdateSamples.push(processingTime)
        if (metrics.positionUpdateSamples.length > MAX_PERFORMANCE_SAMPLES) {
          metrics.positionUpdateSamples.shift()
        }
        metrics.avgPositionUpdateTime = 
          metrics.positionUpdateSamples.reduce((a, b) => a + b, 0) / 
          metrics.positionUpdateSamples.length
      }
    },

    send(data: any) {
      if (!this.service) {
        console.error('[WebSocketStore] Cannot send message: WebSocket service not initialized')
        return
      }
      console.debug('[WebSocketStore] Sending message:', {
        type: data.type,
        timestamp: new Date().toISOString()
      })
      this.messageCount++
      this.lastMessageTime = Date.now()
      this.service.send(data)
    },

    requestInitialData() {
      console.debug('[WebSocketStore] Requesting initial graph data')
      this.initialDataRequested = true
      this.send({ type: 'initial_data' })
    },

    async reconnect() {
      console.debug('[WebSocketStore] Attempting reconnection')
      if (this.service) {
        this.service.cleanup()
        this.service = null
      }
      await this.initialize()
    },

    cleanup() {
      console.debug('[WebSocketStore] Cleaning up websocket store')
      if (this.service) {
        this.service.cleanup()
        this.service = null
      }
      this.connected = false
      this.error = null
      this.lastMessageTime = 0
      this.messageCount = 0
      this.queueSize = 0
      this.connectionAttempts = 0
      this.lastReconnectTime = 0
      this.gpuEnabled = false
      this.initialDataRequested = false
      this.performanceMetrics = {
        avgMessageProcessingTime: 0,
        messageProcessingSamples: [],
        avgPositionUpdateTime: 0,
        positionUpdateSamples: [],
        lastPerformanceReset: Date.now()
      }
    }
  }
})
