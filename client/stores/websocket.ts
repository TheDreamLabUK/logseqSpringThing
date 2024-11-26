import { defineStore } from 'pinia'
import WebsocketService from '../services/websocketService'
import type { BaseMessage, ErrorMessage } from '../types/websocket'

interface WebSocketState {
  connected: boolean
  error: string | null
  service: WebsocketService | null
}

export const useWebSocketStore = defineStore('websocket', {
  state: (): WebSocketState => ({
    connected: false,
    error: null,
    service: null
  }),

  actions: {
    async initialize() {
      if (this.service) {
        console.log('WebSocket service already initialized')
        return
      }

      this.service = new WebsocketService()
      
      // Set up event handlers
      this.service.on('open', () => {
        this.connected = true
        this.error = null
      })

      this.service.on('close', () => {
        this.connected = false
      })

      this.service.on('error', (error: ErrorMessage) => {
        this.error = error.message || 'Unknown error'
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
      this.send({ type: 'getInitialData' })
    },

    cleanup() {
      if (this.service) {
        this.service.cleanup()
        this.service = null
      }
      this.connected = false
      this.error = null
    }
  }
})
