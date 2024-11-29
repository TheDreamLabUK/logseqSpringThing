import type {
  WebSocketConfig,
  WebSocketEventMap,
  WebSocketEventCallback,
  BaseMessage,
  ErrorMessage,
  GraphUpdateMessage,
  BinaryMessage
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

// Constants for heartbeat
const HEARTBEAT_INTERVAL = 30000;
const HEARTBEAT_TIMEOUT = 5000;

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
  
  // Heartbeat properties
  private heartbeatInterval: number | null = null;
  private heartbeatTimeout: number | null = null;
  private lastPongTime: number = 0;

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
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

        const connectionTimeout = setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            console.error('WebSocket connection timeout');
            this.ws?.close();
            reject(new Error('WebSocket connection timeout'));
          }
        }, 10000);

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          console.debug('WebSocket connection established');
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.emit('open');
          this.processQueuedMessages();
          resolve();
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          this.stopHeartbeat();
          console.debug('WebSocket connection closed:', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
          });
          this.handleConnectionClose(event.wasClean);
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

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatInterval = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
        
        this.heartbeatTimeout = window.setTimeout(() => {
          if (Date.now() - this.lastPongTime > HEARTBEAT_TIMEOUT) {
            console.warn('Heartbeat timeout - connection may be dead');
            this.ws?.close();
          }
        }, HEARTBEAT_TIMEOUT);
      }
    }, HEARTBEAT_INTERVAL);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval !== null) {
      window.clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    if (this.heartbeatTimeout !== null) {
      window.clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = null;
    }
  }

  private handleMessage(event: MessageEvent): void {
    try {
      if (event.data instanceof ArrayBuffer) {
        // First float32 is isInitialLayout flag
        const dataView = new Float32Array(event.data);
        const isInitialLayout = dataView[0] >= 1.0;
        
        // Calculate node count from buffer size
        // Buffer contains: isInitialLayout(1) + (x,y,z,vx,vy,vz)(6) per node
        const totalFloats = dataView.length - 1; // Subtract isInitialLayout flag
        const nodeCount = totalFloats / 6; // 6 floats per node
        
        // Emit binary data
        const binaryMessage: BinaryMessage = {
          data: event.data,
          isInitialLayout,
          nodeCount
        };
        
        this.emit('gpuPositions', binaryMessage);
        
      } else {
        // Handle JSON message
        const message = JSON.parse(event.data);
        
        if (message.type === 'pong') {
          this.lastPongTime = Date.now();
          return;
        }

        // Handle graph metadata updates
        if (message.type === 'graphUpdate' || message.type === 'graphData') {
          const graphMessage = message as GraphUpdateMessage;
          this.emit('graphUpdate', graphMessage);
        } else if (message.type === 'error') {
          this.emit('error', message as ErrorMessage);
        }

        this.emit('message', message);
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

  private handleConnectionClose(wasClean: boolean): void {
    this.emit('close');
    
    if (!wasClean && this.reconnectAttempts < this.config.maxRetries) {
      this.reconnectAttempts++;
      const delay = this.calculateReconnectDelay();
      console.debug(`Connection failed. Retrying in ${delay}ms...`);
      
      this.reconnectTimeout = window.setTimeout(() => {
        this.connect().catch(error => {
          console.error('Reconnection attempt failed:', error);
        });
      }, delay);
    } else if (!wasClean) {
      console.error('Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
    }
  }

  private calculateReconnectDelay(): number {
    const baseDelay = this.config.retryDelay;
    const exponentialDelay = baseDelay * Math.pow(2, this.reconnectAttempts - 1);
    const jitter = Math.random() * 1000; // Add up to 1 second of jitter
    return Math.min(exponentialDelay + jitter, 30000); // Cap at 30 seconds
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
    this.stopHeartbeat();
    
    if (this.reconnectTimeout !== null) {
      window.clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }

    this.messageQueue = [];
    this.messageCount = 0;
    this.lastMessageTime = 0;
    this.reconnectAttempts = 0;
    this.eventListeners.clear();
  }
}
