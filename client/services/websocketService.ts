import type {
  WebSocketConfig,
  WebSocketEventMap,
  WebSocketEventCallback,
  BaseMessage,
  ErrorMessage,
  BinaryMessage,
  GraphUpdateMessage,
  NodePosition,
  InitialDataMessage,
  Node
} from '../types/websocket';

import { processPositionUpdate } from '../utils/gpuUtils';

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
  CONNECTION_TIMEOUT,
  UPDATE_THROTTLE_MS
} from '../constants/websocket';

// Add performance monitoring
const measurePerformance = (label: string, fn: () => void) => {
  const start = performance.now();
  fn();
  const duration = performance.now() - start;
  debugLog(`Performance [${label}]`, {
    duration: `${duration.toFixed(2)}ms`
  });
};

// Enhanced debug logging with data validation
const debugLog = (message: string, data?: any) => {
  const timestamp = new Date().toISOString();
  const logPrefix = `[WebsocketService ${timestamp}]`;
  
  console.debug(`${logPrefix} ${message}`);
  
  if (data) {
    if (data instanceof ArrayBuffer) {
      const nodeCount = data.byteLength / 24; // 24 bytes per node (position + velocity)
      const dataView = new Float32Array(data);
      
      console.debug(`${logPrefix} Binary Data Analysis:
        Node Count: ${nodeCount}
        Total Size: ${data.byteLength} bytes
        Expected Size: ${nodeCount * 24} bytes
        Sample Node Data: ${Array.from(dataView.slice(0, 6)).map(v => v.toFixed(2))}`);
    } else if (data instanceof Event) {
      const eventDetails = {
        type: data.type,
        timeStamp: data.timeStamp,
        isTrusted: data.isTrusted,
        target: data instanceof CloseEvent ? {
          code: data.code,
          reason: data.reason,
          wasClean: data.wasClean
        } : undefined,
        readyState: (data.target as WebSocket)?.readyState,
        url: (data.target as WebSocket)?.url
      };
      console.debug(`${logPrefix} Event Details:`, eventDetails);
    } else if (data instanceof Error) {
      console.debug(`${logPrefix} Error Details:`, {
        name: data.name,
        message: data.message,
        stack: data.stack,
        cause: data.cause
      });
    } else if (typeof data === 'object' && data.type === 'initialData') {
      // Special handling for initialData messages
      console.debug(`${logPrefix} Initial Data Message:`, {
        type: data.type,
        graphData: {
          nodeCount: data.graphData?.nodes?.length || 0,
          edgeCount: data.graphData?.edges?.length || 0,
          sampleNodes: data.graphData?.nodes?.slice(0, 3).map((n: { id: string; position?: [number, number, number]; label?: string }) => ({
            id: n.id,
            hasPosition: !!n.position,
            position: n.position,
            label: n.label
          })),
          sampleEdges: data.graphData?.edges?.slice(0, 3).map((e: { source: string; target: string }) => ({
            source: e.source,
            target: e.target
          }))
        },
        hasSettings: !!data.settings,
        timestamp: new Date().toISOString()
      });
    } else if (typeof data === 'object' && data.type === 'graphUpdate') {
      // Special handling for graphUpdate messages
      console.debug(`${logPrefix} Graph Update Message:`, {
        type: data.type,
        graphData: {
          nodeCount: data.graphData?.nodes?.length || 0,
          edgeCount: data.graphData?.edges?.length || 0,
          sampleNodes: data.graphData?.nodes?.slice(0, 3).map((n: { id: string; position?: [number, number, number]; label?: string }) => ({
            id: n.id,
            hasPosition: !!n.position,
            position: n.position,
            label: n.label
          })),
          sampleEdges: data.graphData?.edges?.slice(0, 3).map((e: { source: string; target: string }) => ({
            source: e.source,
            target: e.target
          }))
        },
        timestamp: new Date().toISOString()
      });
    } else if (typeof data === 'object') {
      console.debug(`${logPrefix} Data (${data?.constructor?.name || typeof data}):`, 
        JSON.stringify(data, null, 2));
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
};

export default class WebsocketService {
    private ws: WebSocket | null = null;
    private config: WebSocketConfig;
    private messageQueue: any[] = [];
    private messageCount = 0;
    private lastMessageTime = 0;
    private lastBinaryUpdateTime = 0;  // Track binary update timing
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
    private pendingBinaryUpdate: boolean = false;
    private pendingBinaryBuffer: ArrayBuffer | null = null;  // Store pending binary update

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

  private startHeartbeat(): void {
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

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      window.clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
      debugLog('Heartbeat stopped');
    }
  }

  private reconnect(): void {
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
              timeout: CONNECTION_TIMEOUT,
              url: this.url
            });
            this.ws?.close();
            reject(new Error('WebSocket connection timeout'));
            this.reconnect();
          }
        }, CONNECTION_TIMEOUT);

        this.ws.onopen = (event: Event) => {
          clearTimeout(connectionTimeout);
          debugLog('WebSocket connection established', {
            readyState: this.ws?.readyState,
            url: this.url,
            event
          });
          this.reconnectAttempts = 0;
          this.lastPongTime = Date.now();
          this.startHeartbeat();
          this.emit('open');
          this.processQueuedMessages();
          resolve();
        };

        this.ws.onclose = (event: CloseEvent) => {
          clearTimeout(connectionTimeout);
          this.stopHeartbeat();
          debugLog('WebSocket connection closed', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean,
            readyState: this.ws?.readyState,
            event
          });
          
          if (!this.forceClose) {
            this.reconnect();
          }
          
          this.emit('close', event);
        };

        this.ws.onerror = (event: Event) => {
          clearTimeout(connectionTimeout);
          debugLog('WebSocket error occurred', {
            readyState: this.ws?.readyState,
            url: this.url,
            event
          });

          // Try to get more error details from the event
          let errorDetails = 'Unknown error';
          if (event instanceof ErrorEvent) {
            errorDetails = event.message;
          } else if ('error' in event && event.error instanceof Error) {
            errorDetails = event.error.message;
          } else if (event instanceof Event && event.target instanceof WebSocket) {
            // Try to get information from WebSocket state
            errorDetails = `WebSocket error (readyState: ${event.target.readyState})`;
          }

          const errorMsg: ErrorMessage = {
            type: 'error',
            message: 'WebSocket connection error',
            details: errorDetails,
            code: 'CONNECTION_ERROR'
          };
          this.emit('error', errorMsg);
          
          if (this.ws?.readyState !== WebSocket.OPEN) {
            reject(new Error(`WebSocket connection failed: ${errorDetails}`));
          }
        };

        this.ws.onmessage = this.handleMessage.bind(this);

      } catch (error) {
        debugLog('Error creating WebSocket connection', {
          error,
          url: this.url
        });
        reject(error);
        this.reconnect();
      }
    });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      measurePerformance('handleMessage', () => {
        debugLog('Received WebSocket message', {
          type: event.type,
          dataType: event.data instanceof ArrayBuffer ? 'ArrayBuffer' : typeof event.data,
          dataSize: event.data?.length || 0
        });

        if (event.data instanceof ArrayBuffer) {
          this.handleBinaryMessage(event.data);
        } else {
          this.handleJsonMessage(event.data);
        }
      });
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

    private handleBinaryMessage(data: ArrayBuffer): void {
        if (!this.pendingBinaryUpdate) {
            debugLog('Received unexpected binary data without type information');
            return;
        }

        // Validate buffer size (24 bytes per node)
        const expectedSize = this.indexToNodeId.length * 24;
        if (data.byteLength !== expectedSize) {
            debugLog('Binary message size mismatch', {
                expected: expectedSize,
                received: data.byteLength,
                nodeCount: this.indexToNodeId.length
            });
            return;
        }

        measurePerformance('processPositionUpdate', () => {
            const result = processPositionUpdate(data);
            if (!result) {
                throw new Error('Failed to process position update');
            }

            if (result.positions.length !== this.indexToNodeId.length) {
                debugLog('Position update node count mismatch', {
                    expected: this.indexToNodeId.length,
                    received: result.positions.length
                });
                return;
            }

            // Validate position values
            const positions = result.positions.map((pos, index) => {
                const nodeId = this.indexToNodeId[index];
                if (!nodeId) {
                    debugLog('Missing node ID mapping', { index });
                    return null;
                }

                // Check for invalid values
                const values = [pos.x, pos.y, pos.z, pos.vx, pos.vy, pos.vz];
                const invalidValues = values.map((v, i) => {
                    if (isNaN(v) || !isFinite(v)) {
                        return {
                            index: i,
                            value: v,
                            type: i < 3 ? 'position' : 'velocity',
                            axis: ['x', 'y', 'z'][i % 3]
                        };
                    }
                    return null;
                }).filter(v => v !== null);

                if (invalidValues.length > 0) {
                    debugLog('Invalid position/velocity values', {
                        nodeId,
                        invalidValues,
                        position: pos
                    });
                    return null;
                }

                return {
                    ...pos,
                    id: nodeId
                };
            }).filter((pos): pos is NodePosition & { id: string } => pos !== null);

            const binaryMessage: BinaryMessage = {
                type: 'binaryPositionUpdate',
                data,
                positions,
                nodeCount: this.indexToNodeId.length
            };

            debugLog('Processed binary message', {
                nodeCount: positions.length,
                samplePositions: positions.slice(0, 3).map(pos => ({
                    id: pos.id,
                    position: [pos.x, pos.y, pos.z],
                    velocity: [pos.vx, pos.vy, pos.vz]
                })),
                processingTime: performance.now()
            });

            this.emit('gpuPositions', binaryMessage);
            this.pendingBinaryUpdate = false;
        });
    }

    private handleJsonMessage(data: string): void {
        measurePerformance('handleJsonMessage', () => {
            const message = JSON.parse(data) as BaseMessage;
            debugLog('Parsed JSON message', message);

            if (message.type === 'pong') {
                debugLog('Received pong');
                this.lastPongTime = Date.now();
                return;
            }

            if (message.type === 'binaryPositionUpdate') {
                this.pendingBinaryUpdate = true;
                return;
            }
            
            if (message.type === 'graphUpdate' || message.type === 'initialData') {
                this.handleGraphMessage(message as GraphUpdateMessage | InitialDataMessage);
            } else {
                this.emit('message', message);
                
                if (message.type === 'error') {
                    debugLog('Received error message', message);
                    this.emit('error', message as ErrorMessage);
                }
            }
        });
    }

  private handleGraphMessage(message: GraphUpdateMessage | InitialDataMessage): void {
    debugLog(`Processing ${message.type}`, {
      hasGraphData: !!message.graphData,
      nodeCount: message.graphData?.nodes?.length || 0,
      edgeCount: message.graphData?.edges?.length || 0
    });

    if (message.graphData?.nodes) {
      measurePerformance('updateNodeMappings', () => {
        this.nodeIdToIndex.clear();
        this.indexToNodeId = [];
        
        message.graphData.nodes.forEach((node: Node, index: number) => {
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
      });
    }
    
    if (message.type === 'initialData') {
      this.emit('initialData', message as InitialDataMessage);
    } else {
      this.emit('graphUpdate', message as GraphUpdateMessage);
    }
    
    this.emit('message', message);
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

        const now = Date.now();
        
        // Rate limit to 60fps (16.67ms)
        if (now - this.lastBinaryUpdateTime < UPDATE_THROTTLE_MS) {
            // Store the latest update if we're throttled
            this.pendingBinaryBuffer = data;
            debugLog('Binary update throttled', {
                timeSinceLastUpdate: now - this.lastBinaryUpdateTime,
                throttleMs: UPDATE_THROTTLE_MS
            });
            return;
        }

        try {
            debugLog('Sending binary data', {
                size: data.byteLength,
                timeSinceLastUpdate: now - this.lastBinaryUpdateTime
            });
            
            this.ws.send(data);
            this.lastBinaryUpdateTime = now;
            this.pendingBinaryBuffer = null;  // Clear any pending update

            // Schedule processing of any pending update
            if (this.pendingBinaryBuffer) {
                setTimeout(() => {
                    if (this.pendingBinaryBuffer) {
                        this.sendBinary(this.pendingBinaryBuffer);
                    }
                }, UPDATE_THROTTLE_MS);
            }
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
