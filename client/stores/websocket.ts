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
  isInitialLayout: boolean
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
    isInitialLayout: true,
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
    isGPUEnabled: (state) => state.gpuEnabled,
    isInitialLayoutPhase: (state) => state.isInitialLayout
  },

  actions: {
    async initialize() {
      if (this.service) {
        console.log('WebSocket service already initialized')
        return
      }

      const visualizationStore = useVisualizationStore()
      const binaryUpdateStore = useBinaryUpdateStore()

      this.service = new WebsocketService()
      
      this._setupEventHandlers(visualizationStore, binaryUpdateStore)
      
      try {
        await this.service.connect()
      } catch (error) {
        this._handleConnectionError(error)
        throw error
      }
    },

    _setupEventHandlers(visualizationStore: any, binaryUpdateStore: any) {
      if (!this.service) return

      this.service.on('open', () => {
        console.debug('WebSocket connected, requesting initial data')
        this.connected = true
        this.error = null
        this.connectionAttempts = 0
        this.requestInitialData()
      })

      this.service.on('close', () => {
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
        
        try {
          this._handleGraphUpdate(message, visualizationStore)
        } catch (error) {
          console.error('Error processing graph update:', error)
        }

        this._updateMessageProcessingMetrics(performance.now() - startTime)
      })

      this.service.on('gpuPositions', (message: BinaryMessage) => {
        const startTime = performance.now()
        
        try {
          this.isInitialLayout = message.isInitialLayout
          binaryUpdateStore.updateFromBinary(message.data, message.isInitialLayout)
        } catch (error) {
          console.error('Error processing position update:', error)
        }

        this._updatePositionUpdateMetrics(performance.now() - startTime)
      })

      // Handle simulation mode changes
      this.service.on('simulationModeSet', (mode: string) => {
        visualizationStore.setSimulationMode(mode)
      })

      // Handle all messages to catch GPU and layout state updates
      this.service.on('message', (message: BaseMessage) => {
        if (message.type === 'gpu_state' && 'enabled' in message) {
          this.gpuEnabled = message.enabled
          console.debug(`GPU acceleration ${message.enabled ? 'enabled' : 'disabled'}`)
        }
        else if (message.type === 'layout_state' && 'isInitial' in message) {
          this.isInitialLayout = message.isInitial
          console.debug(`Layout phase: ${message.isInitial ? 'initial' : 'dynamic'}`)
        }
      })
    },

    _handleGraphUpdate(message: GraphUpdateMessage, visualizationStore: any) {
      console.debug('Received graph update:', {
        nodeCount: message.graphData?.nodes?.length || 0,
        edgeCount: message.graphData?.edges?.length || 0,
        timestamp: new Date().toISOString()
      })

      if (!message.graphData && !message.graph_data) {
        console.warn('No graph data found in message')
        return
      }

      const graphData = message.graphData || message.graph_data
      if (!graphData) return

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
      this.error = error instanceof Error ? error.message : 'Unknown connection error'
      this.connectionAttempts++
      this.lastReconnectTime = Date.now()
    },

    _handleError(error: ErrorMessage) {
      this.error = error.message || 'Unknown error'
      if (error.code) {
        console.error(`WebSocket error [${error.code}]:`, this.error)
      } else {
        console.error('WebSocket error:', this.error)
      }
      if (error.details) {
        console.debug('Error details:', error.details)
      }
    },

    _handleDisconnect() {
      const timeSinceLastReconnect = Date.now() - this.lastReconnectTime
      
      if (timeSinceLastReconnect > 60000) { // Reset counter if more than 1 minute since last reconnect
        this.connectionAttempts = 0
      }
      
      this.connectionAttempts++
    },

    _handleMaxReconnectAttempts() {
      this.error = 'Maximum reconnection attempts reached. Please refresh the page.'
      console.error('WebSocket max reconnection attempts reached')
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
        console.error('Cannot send message: WebSocket service not initialized')
        return
      }
      this.messageCount++
      this.lastMessageTime = Date.now()
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
      this.connectionAttempts = 0
      this.lastReconnectTime = 0
      this.gpuEnabled = false
      this.isInitialLayout = true
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
