import type {
  WebSocketConfig,
  WebSocketEventMap,
  WebSocketEventCallback,
  BaseMessage,
  ErrorMessage,
  BinaryMessage,
  GraphUpdateMessage
} from '../types/websocket'

const DEFAULT_CONFIG: WebSocketConfig = {
  messageRateLimit: 60,
  messageTimeWindow: 1000,
  maxMessageSize: 1024 * 1024 * 5, // 5MB
  maxAudioSize: 1024 * 1024 * 10, // 10MB
  maxQueueSize: 1000,
  maxRetries: 3,
  retryDelay: 5000
}

export default class WebsocketService {
  private ws: WebSocket | null = null
  private config: WebSocketConfig
  private messageQueue: any[] = []
  private messageCount = 0
  private lastMessageTime = 0
  private reconnectAttempts = 0
  private reconnectTimeout: number | null = null
  private eventListeners: Map<keyof WebSocketEventMap, Set<WebSocketEventCallback<any>>> = new Map()
  private url: string

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config }
    
    // Use the same host but with /ws path for WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.hostname
    const port = window.location.port ? `:${window.location.port}` : ''
    this.url = `${protocol}//${host}${port}/ws`

    // Log the constructed URL for debugging
    console.debug('WebSocket URL:', this.url)
  }

  public async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.debug('WebSocket already connected')
      return
    }

    return new Promise((resolve, reject) => {
      try {
        console.debug(`Attempting WebSocket connection (attempt ${this.reconnectAttempts + 1}/${this.config.maxRetries})...`, { url: this.url })
        
        this.ws = new WebSocket(this.url)
        
        this.ws.binaryType = 'arraybuffer'

        this.ws.onopen = () => {
          console.debug('WebSocket connection established')
          this.reconnectAttempts = 0
          this.emit('open')
          resolve()
        }

        this.ws.onclose = (event) => {
          console.debug('WebSocket connection closed:', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
          })
          this.handleConnectionClose()
        }

        this.ws.onerror = (error) => {
          console.error('WebSocket connection error:', {
            error,
            url: this.url,
            readyState: this.ws?.readyState
          })
          const errorMsg: ErrorMessage = {
            type: 'error',
            message: 'WebSocket connection error'
          }
          this.emit('error', errorMsg)
          reject(error)
        }

        this.ws.onmessage = (event) => {
          this.handleMessage(event)
        }

      } catch (error) {
        console.error('Error creating WebSocket connection:', {
          error,
          url: this.url
        })
        reject(error)
      }
    })
  }

  private handleConnectionClose(): void {
    this.emit('close')
    
    if (this.reconnectAttempts < this.config.maxRetries) {
      this.reconnectAttempts++
      console.debug(`Connection failed. Retrying in ${this.config.retryDelay}ms...`, {
        attempt: this.reconnectAttempts,
        maxRetries: this.config.maxRetries,
        url: this.url
      })
      
      this.reconnectTimeout = window.setTimeout(() => {
        this.connect().catch(error => {
          console.error('Reconnection attempt failed:', error)
        })
      }, this.config.retryDelay)
    } else {
      console.error('Max reconnection attempts reached', {
        attempts: this.reconnectAttempts,
        url: this.url
      })
      this.emit('maxReconnectAttemptsReached')
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Max reconnection attempts reached'
      }
      this.emit('error', errorMsg)
    }
  }

  private handleMessage(event: MessageEvent): void {
    try {
      if (event.data instanceof ArrayBuffer) {
        // Handle binary message (e.g., position updates)
        const view = new DataView(event.data)
        const isInitialLayout = view.getUint8(0) === 1
        const timeStep = view.getFloat32(1, true)
        const numPositions = (event.data.byteLength - 5) / 24 // 24 bytes per position (6 floats * 4 bytes)
        
        const positions = []
        let offset = 5
        
        for (let i = 0; i < numPositions; i++) {
          positions.push({
            x: view.getFloat32(offset, true),
            y: view.getFloat32(offset + 4, true),
            z: view.getFloat32(offset + 8, true),
            vx: view.getFloat32(offset + 12, true),
            vy: view.getFloat32(offset + 16, true),
            vz: view.getFloat32(offset + 20, true)
          })
          offset += 24
        }

        const binaryMessage: BinaryMessage = {
          isInitialLayout,
          timeStep,
          positions
        }

        this.emit('gpuPositions', binaryMessage)
      } else {
        // Handle JSON message
        const message: BaseMessage = JSON.parse(event.data)
        this.emit('message', message)

        // Emit specific event based on message type
        switch (message.type) {
          case 'graphUpdate':
          case 'graphData':
            this.emit('graphUpdate', message as GraphUpdateMessage)
            break
          case 'ragflowResponse':
            if ('answer' in message) {
              this.emit('ragflowAnswer', message.answer as string)
            }
            break
          case 'simulationModeSet':
            if ('mode' in message) {
              this.emit('simulationModeSet', message.mode as string)
            }
            break
          case 'completion':
            if ('result' in message) {
              this.emit('completion', message.result as string)
            }
            break
          case 'position_update_complete':
            this.emit('positionUpdateComplete', 'complete')
            break
          case 'error':
            if ('message' in message) {
              this.emit('error', message as ErrorMessage)
            }
            break
        }
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error)
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error processing message'
      }
      this.emit('error', errorMsg)
    }
  }

  public send(data: any): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      if (this.messageQueue.length < this.config.maxQueueSize) {
        this.messageQueue.push(data)
      } else {
        console.warn('Message queue full, dropping message')
      }
      return
    }

    const now = Date.now()
    if (now - this.lastMessageTime > this.config.messageTimeWindow) {
      this.messageCount = 0
      this.lastMessageTime = now
    }

    if (this.messageCount >= this.config.messageRateLimit) {
      if (this.messageQueue.length < this.config.maxQueueSize) {
        this.messageQueue.push(data)
      }
      return
    }

    try {
      const message = JSON.stringify(data)
      if (message.length > this.config.maxMessageSize) {
        throw new Error('Message exceeds maximum size')
      }
      
      this.ws.send(message)
      this.messageCount++
      this.lastMessageTime = now

      // Process queued messages
      this.processQueue()
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error sending message'
      }
      this.emit('error', errorMsg)
    }
  }

  private processQueue(): void {
    while (
      this.messageQueue.length > 0 &&
      this.messageCount < this.config.messageRateLimit
    ) {
      const data = this.messageQueue.shift()
      if (data) {
        this.send(data)
      }
    }
  }

  public on<K extends keyof WebSocketEventMap>(
    event: K,
    callback: WebSocketEventCallback<WebSocketEventMap[K]>
  ): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set())
    }
    this.eventListeners.get(event)!.add(callback)
  }

  public off<K extends keyof WebSocketEventMap>(
    event: K,
    callback: WebSocketEventCallback<WebSocketEventMap[K]>
  ): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.delete(callback)
    }
  }

  private emit<K extends keyof WebSocketEventMap>(
    event: K,
    data?: WebSocketEventMap[K]
  ): void {
    const listeners = this.eventListeners.get(event)
    if (listeners) {
      listeners.forEach(callback => callback(data))
    }
  }

  public cleanup(): void {
    if (this.reconnectTimeout !== null) {
      window.clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }

    if (this.ws) {
      this.ws.onclose = null // Prevent reconnection attempt
      this.ws.close()
      this.ws = null
    }

    this.messageQueue = []
    this.messageCount = 0
    this.lastMessageTime = 0
    this.reconnectAttempts = 0
    this.eventListeners.clear()
  }
}
