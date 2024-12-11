import type {
  WebSocketConfig,
  WebSocketEventMap,
  WebSocketEventCallback,
  BaseMessage,
  ErrorMessage,
  BinaryMessage,
  GraphUpdateMessage,
  InitialDataMessage,
  Node
} from '../types/websocket';

import { DebugLogger } from '../utils/validation';

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
  UPDATE_THROTTLE_MS,
  BINARY_UPDATE_NODE_SIZE,
  ERROR_CODES
} from '../constants/websocket';

const DEFAULT_CONFIG: WebSocketConfig = {
  messageRateLimit: DEFAULT_MESSAGE_RATE_LIMIT,
  messageTimeWindow: DEFAULT_MESSAGE_TIME_WINDOW,
  maxMessageSize: DEFAULT_MAX_MESSAGE_SIZE,
  maxAudioSize: DEFAULT_MAX_AUDIO_SIZE,
  maxQueueSize: DEFAULT_MAX_QUEUE_SIZE,
  maxRetries: DEFAULT_RECONNECT_ATTEMPTS,
  retryDelay: DEFAULT_RECONNECT_DELAY
};

const logger = DebugLogger.getInstance();
// Configure logger with settings from settings.toml
logger.configure({
    client_debug: {
        enabled: true,
        enable_websocket_debug: true,
        enable_data_debug: false,
        log_binary_headers: false,
        log_full_json: false
    }
});

// Rate limiting helper
class UpdateThrottler {
  private lastUpdateTime: number = 0;
  private pendingUpdate: ArrayBuffer | null = null;
  private timeoutId: number | null = null;

  constructor(private minInterval: number = UPDATE_THROTTLE_MS) {}

  addUpdate(data: ArrayBuffer): void {
    this.pendingUpdate = data;
    
    if (this.timeoutId === null) {
      const now = performance.now();
      const timeSinceLastUpdate = now - this.lastUpdateTime;
      
      if (timeSinceLastUpdate >= this.minInterval) {
        this.processPendingUpdate();
      } else {
        this.timeoutId = window.setTimeout(
          () => this.processPendingUpdate(),
          this.minInterval - timeSinceLastUpdate
        );
      }
    }
  }

  private processPendingUpdate(): void {
    if (this.pendingUpdate && this.onUpdate) {
      this.onUpdate(this.pendingUpdate);
      this.lastUpdateTime = performance.now();
    }
    this.pendingUpdate = null;
    this.timeoutId = null;
  }

  onUpdate: ((data: ArrayBuffer) => void) | null = null;
}

export default class WebSocketService {
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
    private pendingBinaryUpdate: boolean = false;
    private updateThrottler: UpdateThrottler;

    constructor(config: Partial<WebSocketConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        const hostname = window.location.host;  // Use host instead of hostname to include port if present
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.url = `${protocol}//${hostname}/wss`;
        logger.log('websocket', `WebSocket URL: ${this.url}`);
        
        this.updateThrottler = new UpdateThrottler();
        this.updateThrottler.onUpdate = (data: ArrayBuffer) => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.send(data);
            }
        };
    }

    public async connect(): Promise<void> {
        if (this.ws?.readyState === WebSocket.OPEN) return;

        return new Promise((resolve, reject) => {
            try {
                logger.log('websocket', 'Attempting to connect...');
                this.ws = new WebSocket(this.url);
                this.ws.binaryType = 'arraybuffer';

                const connectionTimeout = setTimeout(() => {
                    if (this.ws?.readyState !== WebSocket.OPEN) {
                        this.ws?.close();
                        logger.log('websocket', 'Connection timeout');
                        reject(new Error('WebSocket connection timeout'));
                        this.reconnect();
                    }
                }, CONNECTION_TIMEOUT);

                this.ws.onopen = () => {
                    clearTimeout(connectionTimeout);
                    this.reconnectAttempts = 0;
                    this.lastPongTime = Date.now();
                    this.startHeartbeat();
                    logger.log('websocket', 'Connection established');
                    this.emit('open');
                    this.processQueuedMessages();
                    resolve();
                };

                this.ws.onclose = (event) => {
                    clearTimeout(connectionTimeout);
                    logger.log('websocket', `Connection closed: ${event.code} - ${event.reason}`);
                    this.handleClose(event);
                };

                this.ws.onerror = (event) => {
                    clearTimeout(connectionTimeout);
                    logger.log('websocket', 'Connection error', event);
                    this.handleError(event);
                    if (this.ws?.readyState !== WebSocket.OPEN) {
                        reject(new Error('WebSocket connection failed'));
                    }
                };

                this.ws.onmessage = this.handleMessage.bind(this);

            } catch (error) {
                logger.log('websocket', 'Connection error:', error);
                reject(error);
                this.reconnect();
            }
        });
    }

    private startHeartbeat(): void {
        if (this.heartbeatInterval) {
            window.clearInterval(this.heartbeatInterval);
        }

        this.heartbeatInterval = window.setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                if (Date.now() - this.lastPongTime > HEARTBEAT_TIMEOUT) {
                    logger.log('websocket', 'Heartbeat timeout, reconnecting...');
                    this.reconnect();
                    return;
                }

                try {
                    this.ws.send(JSON.stringify({ type: 'ping' }));
                    logger.log('websocket', 'Ping sent');
                } catch (error) {
                    logger.log('websocket', 'Error sending ping:', error);
                    this.reconnect();
                }
            }
        }, HEARTBEAT_INTERVAL);
    }

    private handleClose(event: CloseEvent): void {
        this.stopHeartbeat();
        if (!this.forceClose) {
            this.reconnect();
        }
        this.emit('close', event);
    }

    private handleError(event: Event): void {
        const errorDetails = event instanceof ErrorEvent ? event.message :
                           'error' in event ? (event.error as Error).message :
                           'Unknown error';

        const errorMsg: ErrorMessage = {
            type: 'error',
            message: 'WebSocket connection error',
            code: ERROR_CODES.CONNECTION_FAILED,
            details: errorDetails
        };
        
        logger.log('websocket', 'Error:', errorMsg);
        this.emit('error', errorMsg);
    }

    private handleMessage(event: MessageEvent): void {
        try {
            if (event.data instanceof ArrayBuffer) {
                logger.log('binary', 'Received binary message', { size: event.data.byteLength });
                this.handleBinaryMessage(event.data);
            } else {
                logger.log('json', 'Received JSON message', event.data);
                this.handleJsonMessage(event.data);
            }
        } catch (error) {
            const errorMsg: ErrorMessage = {
                type: 'error',
                message: 'Error processing message',
                code: ERROR_CODES.INVALID_MESSAGE,
                details: error instanceof Error ? error.message : String(error)
            };
            logger.log('websocket', 'Message handling error:', errorMsg);
            this.emit('error', errorMsg);
        }
    }

    private handleBinaryMessage(data: ArrayBuffer): void {
        if (!this.pendingBinaryUpdate) return;

        // Validate buffer size
        const expectedSize = this.indexToNodeId.length * BINARY_UPDATE_NODE_SIZE;
        if (data.byteLength !== expectedSize) {
            logger.log('binary', 'Invalid binary message size', {
                expected: expectedSize,
                received: data.byteLength
            });
            return;
        }

        // Forward binary data directly to subscribers
        const binaryMessage: BinaryMessage = {
            type: 'binaryPositionUpdate',
            data
        };

        this.emit('gpuPositions', binaryMessage);
        this.pendingBinaryUpdate = false;
    }

    private handleJsonMessage(data: string): void {
        const message = JSON.parse(data) as BaseMessage;
        
        if (message.type === 'pong') {
            this.lastPongTime = Date.now();
            logger.log('websocket', 'Pong received');
            return;
        }

        if (message.type === 'binaryPositionUpdate') {
            this.pendingBinaryUpdate = true;
            return;
        }

        this.emit('message', message);
        
        switch (message.type) {
            case 'error':
                this.emit('error', message as ErrorMessage);
                break;
            case 'graphUpdate':
                this.handleGraphUpdate(message as GraphUpdateMessage);
                break;
            case 'initialData':
                this.handleInitialData(message as InitialDataMessage);
                break;
            case 'simulationModeSet':
                this.emit('simulationModeSet', message.mode);
                break;
        }
    }

    private handleGraphUpdate(message: GraphUpdateMessage): void {
        if (!message.graphData) return;
        this.updateNodeMappings(message.graphData.nodes);
        this.emit('graphUpdate', message);
    }

    private handleInitialData(message: InitialDataMessage): void {
        if (!message.graphData) return;
        this.updateNodeMappings(message.graphData.nodes);
        this.emit('initialData', message);
    }

    private updateNodeMappings(nodes: Node[]): void {
        this.nodeIdToIndex.clear();
        this.indexToNodeId = [];
        
        nodes.forEach((node, index) => {
            this.nodeIdToIndex.set(node.id, index);
            this.indexToNodeId[index] = node.id;
        });
    }

    public send(data: any): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            if (this.messageQueue.length < this.config.maxQueueSize) {
                this.messageQueue.push(data);
                logger.log('websocket', 'Message queued', data);
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
                logger.log('websocket', 'Message rate limited, queued', data);
            }
            return;
        }

        try {
            const message = JSON.stringify(data);
            if (message.length > this.config.maxMessageSize) {
                throw new Error('Message exceeds maximum size');
            }
            
            this.ws.send(message);
            logger.log('websocket', 'Message sent', data);
            this.messageCount++;
            this.lastMessageTime = now;
            this.processQueue();
        } catch (error) {
            const errorMsg: ErrorMessage = {
                type: 'error',
                message: 'Error sending message',
                code: ERROR_CODES.MESSAGE_TOO_LARGE,
                details: error instanceof Error ? error.message : String(error)
            };
            logger.log('websocket', 'Send error:', errorMsg);
            this.emit('error', errorMsg);
        }
    }

    public sendBinary(data: ArrayBuffer): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        // Validate data size
        const expectedSize = this.indexToNodeId.length * BINARY_UPDATE_NODE_SIZE;
        if (data.byteLength !== expectedSize) {
            logger.log('binary', 'Invalid binary data size', {
                expected: expectedSize,
                received: data.byteLength
            });
            return;
        }

        // Add to throttled updates
        this.updateThrottler.addUpdate(data);
    }

    private processQueue(): void {
        while (
            this.messageQueue.length > 0 &&
            this.messageCount < this.config.messageRateLimit
        ) {
            const data = this.messageQueue.shift();
            if (data) this.send(data);
        }
    }

    private processQueuedMessages(): void {
        const messages = [...this.messageQueue];
        this.messageQueue = [];
        messages.forEach(message => this.send(message));
    }

    public on<K extends keyof WebSocketEventMap>(
        event: K,
        callback: WebSocketEventCallback<WebSocketEventMap[K]>
    ): void {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event)!.add(callback);
        logger.log('websocket', `Event listener added for: ${event}`);
    }

    public off<K extends keyof WebSocketEventMap>(
        event: K,
        callback: WebSocketEventCallback<WebSocketEventMap[K]>
    ): void {
        const listeners = this.eventListeners.get(event);
        if (listeners) {
            listeners.delete(callback);
            logger.log('websocket', `Event listener removed for: ${event}`);
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

    private reconnect(): void {
        if (this.isReconnecting || this.forceClose) return;
        
        this.isReconnecting = true;
        this.cleanup(false);
        
        if (this.reconnectAttempts < this.config.maxRetries) {
            this.reconnectAttempts++;
            const delay = this.config.retryDelay * Math.pow(2, this.reconnectAttempts - 1);
            logger.log('websocket', `Attempting reconnect ${this.reconnectAttempts}/${this.config.maxRetries} in ${delay}ms`);
            
            this.reconnectTimeout = window.setTimeout(() => {
                this.connect().catch(() => {
                    this.reconnect();
                }).finally(() => {
                    this.isReconnecting = false;
                });
            }, delay);
        } else {
            logger.log('websocket', 'Max reconnection attempts reached');
            this.emit('maxReconnectAttemptsReached');
            this.isReconnecting = false;
        }
    }

    private stopHeartbeat(): void {
        if (this.heartbeatInterval) {
            window.clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    public cleanup(force: boolean = true): void {
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
            logger.log('websocket', 'WebSocket connection cleaned up');
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
