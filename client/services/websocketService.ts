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
  HEARTBEAT_TIMEOUT,
  CONNECTION_TIMEOUT
} from '../constants/websocket'

// Debug utility function
const debugLog = (message: string, data?: any) => {
  // Check if debug mode is enabled via environment variable
  if (process.env.DEBUG_MODE === 'true') {
    const timestamp = new Date().toISOString();
    console.debug(`[WebsocketService ${timestamp}] ${message}`);
    
    if (data) {
      if (data instanceof ArrayBuffer) {
        // For binary data, show header info
        const view = new DataView(data);
        const isInitial = view.getFloat32(0, true);
        const nodeCount = (data.byteLength - 4) / 24; // 24 bytes per node (6 float32s)
        console.debug(`Binary Data Header:
          Is Initial: ${isInitial}
          Node Count: ${nodeCount}
          Total Size: ${data.byteLength} bytes`);
      } else if (typeof data === 'object') {
        // For JSON data, show full structure
        console.debug('Data:', JSON.stringify(data, null, 2));
      } else {
        console.debug('Data:', data);
      }
    }
  }
};

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
  private isReconnecting: boolean = false;
  private forceClose: boolean = false;

  constructor(config: Partial<WebSocketConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const hostname = window.location.hostname;
    this.url = `${protocol}//${hostname}/ws`;
    
    debugLog('Initialized with URL:', this.url);
  }

  private startHeartbeat() {
    if (this.heartbeatInterval) {
      window.clearInterval(this.heartbeatInterval);
    }

    this.heartbeatInterval = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        if (Date.now() - this.lastPongTime > HEARTBEAT_TIMEOUT) {
          debugLog('Heartbeat timeout - no pong received');
          this.reconnect();
          return;
        }

        try {
          debugLog('Sending ping');
          this.ws.send(JSON.stringify({ type: 'ping' }));
        } catch (error) {
          debugLog('Error sending heartbeat:', error);
          this.reconnect();
        }
      }
    }, HEARTBEAT_INTERVAL);
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      window.clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private reconnect() {
    if (this.isReconnecting || this.forceClose) return;
    
    this.isReconnecting = true;
    this.cleanup(false);
    
    if (this.reconnectAttempts < this.config.maxRetries) {
      this.reconnectAttempts++;
      const delay = this.config.retryDelay * Math.pow(2, this.reconnectAttempts - 1);
      debugLog(`Connection failed. Retrying in ${delay}ms...`);
      
      this.reconnectTimeout = window.setTimeout(() => {
        this.connect().catch(error => {
          debugLog('Reconnection attempt failed:', error);
          this.reconnect();
        }).finally(() => {
          this.isReconnecting = false;
        });
      }, delay);
    } else {
      debugLog('Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
      this.isReconnecting = false;
    }
  }

  public async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      debugLog('WebSocket already connected');
      return;
    }

    return new Promise((resolve, reject) => {
      try {
        debugLog(`Attempting WebSocket connection (attempt ${this.reconnectAttempts + 1}/${this.config.maxRetries})`);
        
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';

        const connectionTimeout = setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            debugLog('WebSocket connection timeout');
            this.ws?.close();
            reject(new Error('WebSocket connection timeout'));
            this.reconnect();
          }
        }, CONNECTION_TIMEOUT);

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          debugLog('WebSocket connection established');
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
          debugLog('WebSocket connection closed:', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
          });
          
          if (!this.forceClose) {
            this.reconnect();
          }
          
          this.emit('close', event);
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          debugLog('WebSocket connection error:', error);
          const errorMsg: ErrorMessage = {
            type: 'error',
            message: 'WebSocket connection error'
          };
          this.emit('error', errorMsg);
          
          if (this.ws?.readyState !== WebSocket.OPEN) {
            reject(error);
          }
        };

        this.ws.onmessage = this.handleMessage.bind(this);

      } catch (error) {
        debugLog('Error creating WebSocket connection:', error);
        reject(error);
        this.reconnect();
      }
    });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      if (event.data instanceof ArrayBuffer) {
        // Handle binary message (position updates)
        debugLog('Received binary message', event.data);

        const result = processPositionUpdate(event.data);
        if (!result) {
          throw new Error('Failed to process position update');
        }

        if (result.positions.length !== this.indexToNodeId.length) {
          debugLog('Position update node count mismatch:', {
            expected: this.indexToNodeId.length,
            received: result.positions.length
          });
        }

        const positions = result.positions.map((pos, index) => {
          const nodeId = this.indexToNodeId[index];
          if (!nodeId) {
            debugLog(`No stored ID for node index ${index}`);
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
        debugLog('Received JSON message', message);

        // Handle pong messages
        if (message.type === 'pong') {
          debugLog('Received pong');
          this.lastPongTime = Date.now();
          return;
        }
        
        // Handle graph updates and store node mappings
        if (message.type === 'graphUpdate') {
          const graphMessage = message as GraphUpdateMessage;
          debugLog('Processing graph update', {
            hasGraphData: !!graphMessage.graphData,
            nodeCount: graphMessage.graphData?.nodes?.length || 0
          });

          if (graphMessage.graphData?.nodes) {
            this.nodeIdToIndex.clear();
            this.indexToNodeId = [];
            
            graphMessage.graphData.nodes.forEach((node, index) => {
              this.nodeIdToIndex.set(node.id, index);
              this.indexToNodeId[index] = node.id;
            });
            
            debugLog('Node ID mappings updated', {
              count: this.indexToNodeId.length,
              sampleIds: this.indexToNodeId.slice(0, 3),
              sampleNodes: graphMessage.graphData.nodes.slice(0, 3).map(n => ({
                id: n.id,
                hasPosition: !!n.position,
                position: n.position
              }))
            });
          }
          
          this.emit('graphUpdate', graphMessage);
        } else {
          this.emit('message', message);
          
          if (message.type === 'error') {
            debugLog('Received error message:', message);
            this.emit('error', message as ErrorMessage);
          }
        }
      }
    } catch (error) {
      debugLog('Error handling WebSocket message:', error);
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error processing message'
      };
      this.emit('error', errorMsg);
    }
  }

  public send(data: any): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      if (this.messageQueue.length < this.config.maxQueueSize) {
        this.messageQueue.push(data);
        debugLog('Message queued', {
          type: data.type,
          queueSize: this.messageQueue.length
        });
      } else {
        debugLog('Message queue full, dropping message');
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
        debugLog('Rate limited, message queued', {
          type: data.type,
          queueSize: this.messageQueue.length
        });
      }
      return;
    }

    try {
      const message = JSON.stringify(data);
      if (message.length > this.config.maxMessageSize) {
        throw new Error('Message exceeds maximum size');
      }
      
      debugLog('Sending message', {
        type: data.type,
        size: message.length
      });

      this.ws.send(message);
      this.messageCount++;
      this.lastMessageTime = now;

      this.processQueue();
    } catch (error) {
      debugLog('Error sending message:', error);
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error sending message'
      };
      this.emit('error', errorMsg);
    }
  }

  public sendBinary(data: ArrayBuffer): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      debugLog('Cannot send binary data: WebSocket not open');
      return;
    }

    try {
      debugLog('Sending binary data', data);
      this.ws.send(data);
    } catch (error) {
      debugLog('Error sending binary data:', error);
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error sending binary data'
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
        debugLog('Processing queued message', {
          type: data.type,
          remainingQueue: this.messageQueue.length
        });
        this.send(data);
      }
    }
  }

  private processQueuedMessages(): void {
    if (this.messageQueue.length > 0) {
      debugLog(`Processing ${this.messageQueue.length} queued messages`);
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

  public cleanup(force: boolean = true): void {
    debugLog('Cleaning up websocket service');
    this.stopHeartbeat();
    
    if (this.reconnectTimeout !== null) {
      window.clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.forceClose = force;
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }

    this.messageQueue = [];
    this.messageCount = 0;
    this.lastMessageTime = 0;
    if (force) {
      this.reconnectAttempts = 0;
      this.eventListeners.clear();
      this.nodeIdToIndex.clear();
      this.indexToNodeId = [];
    }
  }
}
