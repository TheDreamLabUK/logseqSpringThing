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

// Enhanced debug logging
const debugLog = (message: string, data?: any) => {
  const timestamp = new Date().toISOString();
  const logPrefix = `[WebsocketService ${timestamp}]`;
  
  // Always log the message
  console.debug(`${logPrefix} ${message}`);
  
  if (data) {
    if (data instanceof ArrayBuffer) {
      // For binary data, show detailed header info
      const view = new DataView(data);
      const isInitial = view.getFloat32(0, true);
      const nodeCount = (data.byteLength - 4) / 24;
      console.debug(`${logPrefix} Binary Data Analysis:
        Is Initial: ${isInitial}
        Node Count: ${nodeCount}
        Total Size: ${data.byteLength} bytes
        Header Bytes: ${Array.from(new Uint8Array(data.slice(0, 16))).map(b => b.toString(16).padStart(2, '0')).join(' ')}`);
    } else if (data instanceof Event) {
      // For WebSocket events, show relevant properties
      const eventDetails = {
        type: data.type,
        timeStamp: data.timeStamp,
        isTrusted: data.isTrusted,
        // Add other known Event properties as needed
        ...(data instanceof CloseEvent ? {
          code: data.code,
          reason: data.reason,
          wasClean: data.wasClean
        } : {}),
        ...(data instanceof MessageEvent ? {
          data: data.data,
          origin: data.origin,
          lastEventId: data.lastEventId
        } : {})
      };
      console.debug(`${logPrefix} Event Details:`, eventDetails);
    } else if (typeof data === 'object') {
      // For JSON data, show full structure with type information
      console.debug(`${logPrefix} Data (${data?.constructor?.name || typeof data}):`, JSON.stringify(data, null, 2));
    } else {
      console.debug(`${logPrefix} Data (${typeof data}):`, data);
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
    
    // Use wss:// for production
    const hostname = window.location.hostname;
    this.url = `wss://${hostname}/ws`;
    
    debugLog('WebSocket service initialized', {
      url: this.url,
      config: this.config,
      protocol: window.location.protocol,
      hostname: hostname
    });
  }

  private startHeartbeat() {
    if (this.heartbeatInterval) {
      window.clearInterval(this.heartbeatInterval);
    }

    this.heartbeatInterval = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        if (Date.now() - this.lastPongTime > HEARTBEAT_TIMEOUT) {
          debugLog('Heartbeat timeout - no pong received', {
            lastPongTime: new Date(this.lastPongTime).toISOString(),
            timeout: HEARTBEAT_TIMEOUT,
            timeSinceLastPong: Date.now() - this.lastPongTime
          });
          this.reconnect();
          return;
        }

        try {
          debugLog('Sending ping');
          this.ws.send(JSON.stringify({ type: 'ping' }));
        } catch (error) {
          debugLog('Error sending heartbeat', error);
          this.reconnect();
        }
      }
    }, HEARTBEAT_INTERVAL);

    debugLog('Heartbeat started', {
      interval: HEARTBEAT_INTERVAL,
      timeout: HEARTBEAT_TIMEOUT
    });
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      window.clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
      debugLog('Heartbeat stopped');
    }
  }

  private reconnect() {
    if (this.isReconnecting || this.forceClose) {
      debugLog('Reconnect skipped', {
        isReconnecting: this.isReconnecting,
        forceClose: this.forceClose
      });
      return;
    }
    
    this.isReconnecting = true;
    this.cleanup(false);
    
    if (this.reconnectAttempts < this.config.maxRetries) {
      this.reconnectAttempts++;
      const delay = this.config.retryDelay * Math.pow(2, this.reconnectAttempts - 1);
      debugLog('Scheduling reconnection', {
        attempt: this.reconnectAttempts,
        maxRetries: this.config.maxRetries,
        delay: delay
      });
      
      this.reconnectTimeout = window.setTimeout(() => {
        this.connect().catch(error => {
          debugLog('Reconnection attempt failed', error);
          this.reconnect();
        }).finally(() => {
          this.isReconnecting = false;
        });
      }, delay);
    } else {
      debugLog('Max reconnection attempts reached', {
        attempts: this.reconnectAttempts,
        maxRetries: this.config.maxRetries
      });
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
        debugLog('Initiating WebSocket connection', {
          attempt: this.reconnectAttempts + 1,
          maxRetries: this.config.maxRetries,
          url: this.url
        });
        
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';

        const connectionTimeout = setTimeout(() => {
          if (this.ws?.readyState !== WebSocket.OPEN) {
            debugLog('Connection timeout', {
              readyState: this.ws?.readyState,
              timeout: CONNECTION_TIMEOUT
            });
            this.ws?.close();
            reject(new Error('WebSocket connection timeout'));
            this.reconnect();
          }
        }, CONNECTION_TIMEOUT);

        this.ws.onopen = () => {
          clearTimeout(connectionTimeout);
          debugLog('WebSocket connection established', {
            readyState: this.ws?.readyState,
            url: this.url
          });
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
          debugLog('WebSocket connection closed', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean,
            timestamp: new Date().toISOString(),
            readyState: this.ws?.readyState
          });
          
          if (!this.forceClose) {
            this.reconnect();
          }
          
          this.emit('close', event);
        };

        this.ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          debugLog('WebSocket error occurred', {
            error: error,
            readyState: this.ws?.readyState,
            url: this.url,
            timestamp: new Date().toISOString()
          });
          const errorMsg: ErrorMessage = {
            type: 'error',
            message: 'WebSocket connection error',
            details: JSON.stringify(error)
          };
          this.emit('error', errorMsg);
          
          if (this.ws?.readyState !== WebSocket.OPEN) {
            reject(error);
          }
        };

        this.ws.onmessage = this.handleMessage.bind(this);

      } catch (error) {
        debugLog('Error creating WebSocket connection', {
          error,
          url: this.url,
          timestamp: new Date().toISOString()
        });
        reject(error);
        this.reconnect();
      }
    });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      debugLog('Received WebSocket message', {
        type: event.type,
        dataType: event.data instanceof ArrayBuffer ? 'ArrayBuffer' : typeof event.data,
        dataSize: event.data?.length || 0
      });

      if (event.data instanceof ArrayBuffer) {
        // Handle binary message (position updates)
        const result = processPositionUpdate(event.data);
        if (!result) {
          throw new Error('Failed to process position update');
        }

        if (result.positions.length !== this.indexToNodeId.length) {
          debugLog('Position update node count mismatch', {
            expected: this.indexToNodeId.length,
            received: result.positions.length
          });
        }

        const positions = result.positions.map((pos, index) => {
          const nodeId = this.indexToNodeId[index];
          if (!nodeId) {
            debugLog('Missing node ID mapping', { index });
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

        debugLog('Processed binary message', {
          nodeCount: positions.length,
          samplePositions: positions.slice(0, 3)
        });

        this.emit('gpuPositions', binaryMessage);

      } else {
        // Handle JSON message
        const message = JSON.parse(event.data) as BaseMessage;
        debugLog('Parsed JSON message', message);

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
            nodeCount: graphMessage.graphData?.nodes?.length || 0,
            sampleNodes: graphMessage.graphData?.nodes?.slice(0, 3)
          });

          if (graphMessage.graphData?.nodes) {
            this.nodeIdToIndex.clear();
            this.indexToNodeId = [];
            
            graphMessage.graphData.nodes.forEach((node, index) => {
              this.nodeIdToIndex.set(node.id, index);
              this.indexToNodeId[index] = node.id;
            });
            
            debugLog('Node ID mappings updated', {
              mappingCount: this.indexToNodeId.length,
              sampleMappings: this.indexToNodeId.slice(0, 3).map(id => ({
                id,
                index: this.nodeIdToIndex.get(id)
              }))
            });
          }
          
          this.emit('graphUpdate', graphMessage);
        } else {
          this.emit('message', message);
          
          if (message.type === 'error') {
            debugLog('Received error message', message);
            this.emit('error', message as ErrorMessage);
          }
        }
      }
    } catch (error) {
      debugLog('Error handling WebSocket message', {
        error,
        eventType: event.type,
        dataType: typeof event.data
      });
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error processing message',
        details: error instanceof Error ? error.message : String(error)
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
          queueSize: this.messageQueue.length,
          maxQueueSize: this.config.maxQueueSize
        });
      } else {
        debugLog('Message queue full, dropping message', {
          type: data.type,
          queueSize: this.messageQueue.length
        });
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
          queueSize: this.messageQueue.length,
          messageCount: this.messageCount,
          rateLimit: this.config.messageRateLimit
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
        size: message.length,
        maxSize: this.config.maxMessageSize
      });

      this.ws.send(message);
      this.messageCount++;
      this.lastMessageTime = now;

      this.processQueue();
    } catch (error) {
      debugLog('Error sending message', {
        error,
        data: data
      });
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error sending message',
        details: error instanceof Error ? error.message : String(error)
      };
      this.emit('error', errorMsg);
    }
  }

  public sendBinary(data: ArrayBuffer): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      debugLog('Cannot send binary data: WebSocket not open', {
        readyState: this.ws?.readyState,
        dataSize: data.byteLength
      });
      return;
    }

    try {
      debugLog('Sending binary data', {
        size: data.byteLength,
        header: Array.from(new Uint8Array(data.slice(0, 16))).map(b => b.toString(16).padStart(2, '0')).join(' ')
      });
      this.ws.send(data);
    } catch (error) {
      debugLog('Error sending binary data', {
        error,
        dataSize: data.byteLength
      });
      const errorMsg: ErrorMessage = {
        type: 'error',
        message: 'Error sending binary data',
        details: error instanceof Error ? error.message : String(error)
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
          remainingQueue: this.messageQueue.length,
          messageCount: this.messageCount
        });
        this.send(data);
      }
    }
  }

  private processQueuedMessages(): void {
    if (this.messageQueue.length > 0) {
      debugLog('Processing queued messages', {
        count: this.messageQueue.length
      });
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
    debugLog('Event listener added', { event });
  }

  public off<K extends keyof WebSocketEventMap>(
    event: K,
    callback: WebSocketEventCallback<WebSocketEventMap[K]>
  ): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(callback);
      debugLog('Event listener removed', { event });
    }
  }

  private emit<K extends keyof WebSocketEventMap>(
    event: K,
    data?: WebSocketEventMap[K]
  ): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      debugLog('Emitting event', {
        event,
        listenerCount: listeners.size,
        data: data
      });
      listeners.forEach(callback => callback(data));
    }
  }

  public cleanup(force: boolean = true): void {
    debugLog('Cleaning up websocket service', {
      force,
      isReconnecting: this.isReconnecting,
      hasActiveConnection: !!this.ws
    });
    
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
