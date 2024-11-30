import type {
  WebSocketConfig,
  WebSocketEventMap,
  WebSocketEventCallback,
  BaseMessage,
  ErrorMessage,
  BinaryMessage,
  GraphUpdateMessage,
  NodePosition
} from '../types/websocket'

import { processPositionUpdate } from '../utils/gpuUtils'

import {
  DEFAULT_RECONNECT_ATTEMPTS,
  DEFAULT_RECONNECT_DELAY,
  DEFAULT_MESSAGE_RATE_LIMIT,
  DEFAULT_MESSAGE_TIME_WINDOW,
  DEFAULT_MAX_MESSAGE_SIZE,
  DEFAULT_MAX_AUDIO_SIZE,
  DEFAULT_MAX_QUEUE_SIZE,
  HEARTBEAT_INTERVAL,
  HEARTBEAT_TIMEOUT
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
  private heartbeatInterval: number | null = null;
  private lastPongTime: number = Date.now();
  private eventListeners: Map<keyof WebSocketEventMap, Set<WebSocketEventCallback<any>>> = new Map();
  private url: string;
  private nodeIdToIndex: Map<string, number> = new Map();
  private indexToNodeId: string[] = [];

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.url = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
    console.debug('[WebsocketService] Initialized with URL:', this.url);
  }

  private startHeartbeat() {
    if (this.heartbeatInterval) {
      window.clearInterval(this.heartbeatInterval);
    }

    this.heartbeatInterval = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        // Check if we've received a pong within the timeout period
        if (Date.now() - this.lastPongTime > HEARTBEAT_TIMEOUT) {
          console.warn('[WebsocketService] Heartbeat timeout - no pong received');
          this.ws.close();
          return;
        }

        // Send ping message
        console.debug('[WebsocketService] Sending ping');
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, HEARTBEAT_INTERVAL);
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      window.clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  public async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.debug('[WebsocketService] WebSocket already connected');
      return;
    }

    return new Promise((resolve, reject) => {
      try {
        console.debug(`[WebsocketService] Attempting WebSocket connection (attempt ${this.reconnectAttempts + 1}/${this.config.maxRetries})...`);
        
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';

        // Set a connection timeout
        const connectionTimeout = setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            console.error('[WebsocketService] WebSocket connection timeout');
            this.ws?.close();
            reject(new Error('WebSocket connection timeout'));
          }
        }, 10000);

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          console.debug('[WebsocketService] WebSocket connection established');
          this.reconnectAttempts = 0;
          this.lastPongTime = Date.now();
          this.startHeartbeat();
          this.emit('open');
          this.processQueuedMessages();
          resolve();
        };

        this.ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          this.stopHeartbeat();
          console.debug('[WebsocketService] WebSocket connection closed:', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
          });
          this.handleConnectionClose();
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          console.error('[WebsocketService] WebSocket connection error:', error);
          const errorMsg: ErrorMessage = {
            type: 'error',
            message: 'WebSocket connection error'
          };
          this.emit('error', errorMsg);
          reject(error);
        };

        this.ws.onmessage = this.handleMessage.bind(this);

      } catch (error) {
        console.error('[WebsocketService] Error creating WebSocket connection:', error);
        reject(error);
      }
    });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      if (event.data instanceof ArrayBuffer) {
        // Handle binary message (position updates)
        console.debug('[WebsocketService] Received binary message:', {
          size: event.data.byteLength,
          expectedSize: this.indexToNodeId.length * 24,
          timestamp: new Date().toISOString()
        });

        const result = processPositionUpdate(event.data);
        if (!result) {
          throw new Error('Failed to process position update');
        }

        if (result.positions.length !== this.indexToNodeId.length) {
          console.warn('[WebsocketService] Position update node count mismatch:', {
            expected: this.indexToNodeId.length,
            received: result.positions.length,
            timestamp: new Date().toISOString()
          });
        }

        const positions = result.positions.map((pos, index) => {
          const nodeId = this.indexToNodeId[index];
          if (!nodeId) {
            console.warn(`[WebsocketService] No stored ID for node index ${index}`);
            return null;
          }
          return {
            ...pos,
            id: nodeId
          };
        }).filter((pos): pos is NodePosition & { id: string } => pos !== null);

        const binaryMessage: BinaryMessage = {
          data: event.data,
          positions,
          nodeCount: this.indexToNodeId.length
        };

        this.emit('gpuPositions', binaryMessage);

      } else {
        // Handle JSON message
        const message = JSON.parse(event.data) as BaseMessage;
        console.debug('[WebsocketService] Received JSON message:', {
          type: message.type,
          timestamp: new Date().toISOString()
        });

        // Handle pong messages
        if (message.type === 'pong') {
          console.debug('[WebsocketService] Received pong');
          this.lastPongTime = Date.now();
          return;
        }
        
        // Handle graph updates and store node mappings
        if (message.type === 'graphUpdate') {
          const graphMessage = message as GraphUpdateMessage;
          console.debug('[WebsocketService] Processing graph update:', {
            hasGraphData: !!graphMessage.graphData,
            hasSnakeCaseData: !!graphMessage.graph_data,
            nodeCount: graphMessage.graphData?.nodes?.length || graphMessage.graph_data?.nodes?.length || 0,
            timestamp: new Date().toISOString()
          });

          const graphData = graphMessage.graphData || graphMessage.graph_data;
          
          if (graphData?.nodes) {
            this.nodeIdToIndex.clear();
            this.indexToNodeId = [];
            
            graphData.nodes.forEach((node, index) => {
              this.nodeIdToIndex.set(node.id, index);
              this.indexToNodeId[index] = node.id;
            });
            
            console.debug('[WebsocketService] Node ID mappings updated:', {
              count: this.indexToNodeId.length,
              sampleIds: this.indexToNodeId.slice(0, 3),
              sampleNodes: graphData.nodes.slice(0, 3).map(n => ({
                id: n.id,
                hasPosition: !!n.position,
                position: n.position
              })),
              timestamp: new Date().toISOString()
            });
          }
          
          this.emit('graphUpdate', graphMessage);
        } else {
          this.emit('message', message);
          
          if (message.type === 'error') {
            console.error('[WebsocketService] Received error message:', message);
            this.emit('error', message as ErrorMessage);
          }
        }
      }
    } catch (error) {
      console.error('[WebsocketService] Error handling WebSocket message:', error);
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
      const delay = this.config.retryDelay * Math.pow(2, this.reconnectAttempts - 1);
      console.debug(`[WebsocketService] Connection failed. Retrying in ${delay}ms...`);
      
      this.reconnectTimeout = window.setTimeout(() => {
        this.connect().catch(error => {
          console.error('[WebsocketService] Reconnection attempt failed:', error);
        });
      }, delay);
    } else {
      console.error('[WebsocketService] Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
    }
  }

  public send(data: any): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      if (this.messageQueue.length < this.config.maxQueueSize) {
        this.messageQueue.push(data);
        console.debug('[WebsocketService] Message queued:', {
          type: data.type,
          queueSize: this.messageQueue.length,
          timestamp: new Date().toISOString()
        });
      } else {
        console.warn('[WebsocketService] Message queue full, dropping message');
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
        console.debug('[WebsocketService] Rate limited, message queued:', {
          type: data.type,
          queueSize: this.messageQueue.length,
          timestamp: new Date().toISOString()
        });
      }
      return;
    }

    try {
      const message = JSON.stringify(data);
      if (message.length > this.config.maxMessageSize) {
        throw new Error('Message exceeds maximum size');
      }
      
      console.debug('[WebsocketService] Sending message:', {
        type: data.type,
        size: message.length,
        timestamp: new Date().toISOString()
      });

      this.ws.send(message);
      this.messageCount++;
      this.lastMessageTime = now;

      this.processQueue();
    } catch (error) {
      console.error('[WebsocketService] Error sending message:', error);
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
        console.debug('[WebsocketService] Processing queued message:', {
          type: data.type,
          remainingQueue: this.messageQueue.length,
          timestamp: new Date().toISOString()
        });
        this.send(data);
      }
    }
  }

  private processQueuedMessages(): void {
    if (this.messageQueue.length > 0) {
      console.debug(`[WebsocketService] Processing ${this.messageQueue.length} queued messages`);
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
    console.debug('[WebsocketService] Cleaning up websocket service');
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
    this.nodeIdToIndex.clear();
    this.indexToNodeId = [];
  }
}
