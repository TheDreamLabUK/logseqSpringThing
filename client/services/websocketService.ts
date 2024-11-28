import type {
  WebSocketConfig,
  WebSocketEventMap,
  WebSocketEventCallback,
  BaseMessage,
  ErrorMessage,
  BinaryMessage,
  GraphUpdateMessage,
  PositionUpdate
} from '../types/websocket'

import {
  DEFAULT_RECONNECT_ATTEMPTS,
  DEFAULT_RECONNECT_DELAY,
  DEFAULT_MESSAGE_RATE_LIMIT,
  DEFAULT_MESSAGE_TIME_WINDOW,
  DEFAULT_MAX_MESSAGE_SIZE,
  DEFAULT_MAX_AUDIO_SIZE,
  DEFAULT_MAX_QUEUE_SIZE
} from '../constants/websocket'

const DEFAULT_CONFIG: WebSocketConfig = {
  messageRateLimit: DEFAULT_MESSAGE_RATE_LIMIT,
  messageTimeWindow: DEFAULT_MESSAGE_TIME_WINDOW,
  maxMessageSize: DEFAULT_MAX_MESSAGE_SIZE,
  maxAudioSize: DEFAULT_MAX_AUDIO_SIZE,
  maxQueueSize: DEFAULT_MAX_QUEUE_SIZE,
  maxRetries: DEFAULT_RECONNECT_ATTEMPTS,
  retryDelay: DEFAULT_RECONNECT_DELAY
}

export default class WebsocketService {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private messageQueue: any[] = [];
  private messageCount = 0;
  private lastMessageTime = 0;
  private reconnectAttempts = 0;
  private reconnectTimeout: number | null = null;
  private eventListeners: Map<keyof WebSocketEventMap, Set<WebSocketEventCallback<any>>> = new Map();
  private url: string;
  // Store node IDs in order they appear in initial graph data
  private nodeIds: string[] = [];

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    // Use relative path for WebSocket to maintain protocol and host
    this.url = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;

    console.debug('WebSocket URL:', this.url);
  }

  public async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.debug('WebSocket already connected');
      return;
    }

    return new Promise((resolve, reject) => {
      try {
        console.debug(`Attempting WebSocket connection (attempt ${this.reconnectAttempts + 1}/${this.config.maxRetries})...`);
        
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';

        // Set a connection timeout
        const connectionTimeout = setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            console.error('WebSocket connection timeout');
            this.ws?.close();
            reject(new Error('WebSocket connection timeout'));
          }
        }, 10000); // 10 second timeout

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          console.debug('WebSocket connection established');
          this.reconnectAttempts = 0;
          this.emit('open');
          this.processQueuedMessages();
          resolve();
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          console.debug('WebSocket connection closed:', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
          });
          this.handleConnectionClose();
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          console.error('WebSocket connection error:', error);
          const errorMsg: ErrorMessage = {
            type: 'error',
            message: 'WebSocket connection error'
          };
          this.emit('error', errorMsg);
          reject(error);
        };

        this.ws.onmessage = this.handleMessage.bind(this);

      } catch (error) {
        console.error('Error creating WebSocket connection:', error);
        reject(error);
      }
    });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      if (event.data instanceof ArrayBuffer) {
        // Handle binary message (position updates)
        const view = new DataView(event.data);
        const isInitialLayout = view.getFloat32(0, true) >= 1.0;
        const numPositions = (event.data.byteLength - 4) / 24; // 24 bytes per position (6 float32s * 4 bytes)
        
        // Log binary update stats
        console.debug('Binary position update:', {
          isInitialLayout,
          numPositions,
          numStoredIds: this.nodeIds.length,
          dataSize: event.data.byteLength
        });
        
        const positions: PositionUpdate[] = [];
        let offset = 4; // Skip isInitialLayout flag
        
        for (let i = 0; i < numPositions; i++) {
          // Use stored node ID for this position index
          const nodeId = this.nodeIds[i];
          if (!nodeId) {
            console.warn(`No stored ID for node index ${i}`);
            continue;
          }
          
          // Read raw float32 values
          const x = view.getFloat32(offset, true);
          const y = view.getFloat32(offset + 4, true);
          const z = view.getFloat32(offset + 8, true);
          const vx = view.getFloat32(offset + 12, true);
          const vy = view.getFloat32(offset + 16, true);
          const vz = view.getFloat32(offset + 20, true);
          
          positions.push({
            id: nodeId,
            x, y, z,
            vx, vy, vz
          });
          offset += 24;
        }

        // Log first few positions for debugging
        if (positions.length > 0) {
          console.debug('Sample positions:', positions.slice(0, 3));
        }

        const binaryMessage: BinaryMessage = {
          isInitialLayout,
          positions
        };

        this.emit('gpuPositions', binaryMessage);
      } else {
        // Handle JSON message
        const message: BaseMessage = JSON.parse(event.data);
        this.emit('message', message);

        // Store node IDs from initial graph data
        if (message.type === 'graphUpdate' || message.type === 'graphData') {
          const graphMessage = message as GraphUpdateMessage;
          if (graphMessage.graphData?.nodes) {
            // Store node IDs in order they appear in the array
            this.nodeIds = graphMessage.graphData.nodes.map(node => node.id);
            console.debug('Stored node IDs:', {
              count: this.nodeIds.length,
              sample: this.nodeIds.slice(0, 3)
            });
          }
          this.emit('graphUpdate', graphMessage);
        } else if (message.type === 'error') {
          this.emit('error', message as ErrorMessage);
        }
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error processing message'
      };
      this.emit('error', errorMsg);
    }
  }

  private handleConnectionClose(): void {
    this.emit('close');
    
    if (this.reconnectAttempts < this.config.maxRetries) {
      this.reconnectAttempts++;
      const delay = this.config.retryDelay * Math.pow(2, this.reconnectAttempts - 1); // Exponential backoff
      console.debug(`Connection failed. Retrying in ${delay}ms...`);
      
      this.reconnectTimeout = window.setTimeout(() => {
        this.connect().catch(error => {
          console.error('Reconnection attempt failed:', error);
        });
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
    }
  }

  public send(data: any): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      if (this.messageQueue.length < this.config.maxQueueSize) {
        this.messageQueue.push(data);
      } else {
        console.warn('Message queue full, dropping message');
      }
      return;
    }

    const now = Date.now();
    if (now - this.lastMessageTime > this.config.messageTimeWindow) {
      this.messageCount = 0;
      this.lastMessageTime = now;
    }

    if (this.messageCount >= this.config.messageRateLimit) {
      if (this.messageQueue.length < this.config.maxQueueSize) {
        this.messageQueue.push(data);
      }
      return;
    }

    try {
      const message = JSON.stringify(data);
      if (message.length > this.config.maxMessageSize) {
        throw new Error('Message exceeds maximum size');
      }
      
      this.ws.send(message);
      this.messageCount++;
      this.lastMessageTime = now;

      // Process queued messages
      this.processQueue();
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error sending message'
      };
      this.emit('error', errorMsg);
    }
  }

  private processQueue(): void {
    while (
      this.messageQueue.length > 0 &&
      this.messageCount < this.config.messageRateLimit
    ) {
      const data = this.messageQueue.shift();
      if (data) {
        this.send(data);
      }
    }
  }

  private processQueuedMessages(): void {
    if (this.messageQueue.length > 0) {
      console.debug(`Processing ${this.messageQueue.length} queued messages`);
      const messages = [...this.messageQueue];
      this.messageQueue = [];
      messages.forEach(message => this.send(message));
    }
  }

  public on<K extends keyof WebSocketEventMap>(
    event: K,
    callback: WebSocketEventCallback<WebSocketEventMap[K]>
  ): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
  }

  public off<K extends keyof WebSocketEventMap>(
    event: K,
    callback: WebSocketEventCallback<WebSocketEventMap[K]>
  ): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(callback);
    }
  }

  private emit<K extends keyof WebSocketEventMap>(
    event: K,
    data?: WebSocketEventMap[K]
  ): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(callback => callback(data));
    }
  }

  public cleanup(): void {
    if (this.reconnectTimeout !== null) {
      window.clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.onclose = null; // Prevent reconnection attempt
      this.ws.close();
      this.ws = null;
    }

    this.messageQueue = [];
    this.messageCount = 0;
    this.lastMessageTime = 0;
    this.reconnectAttempts = 0;
    this.eventListeners.clear();
    this.nodeIds = []; // Clear stored node IDs
  }
}
