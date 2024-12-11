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

// Performance monitoring constants
const PERFORMANCE_SAMPLE_SIZE = 60; // 1 second worth at 60fps
const TARGET_FRAME_TIME = 16.67; // Target 60fps
const MIN_UPDATE_INTERVAL = 16.67; // Max 60fps
const MAX_UPDATE_INTERVAL = 200; // Min 5fps
const INTERACTION_UPDATE_INTERVAL = 33.33; // 30fps during interaction

class PerformanceMonitor {
  private frameTimes: number[] = [];
  private lastFrameTime: number = 0;
  private currentUpdateInterval: number = MIN_UPDATE_INTERVAL;

  addFrameTime(timestamp: number) {
    const frameTime = timestamp - this.lastFrameTime;
    this.lastFrameTime = timestamp;

    if (frameTime > 0) {
      this.frameTimes.push(frameTime);
      if (this.frameTimes.length > PERFORMANCE_SAMPLE_SIZE) {
        this.frameTimes.shift();
      }
    }
  }

  getAverageFrameTime(): number {
    if (this.frameTimes.length === 0) return TARGET_FRAME_TIME;
    return this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
  }

  calculateUpdateInterval(isInteracting: boolean): number {
    const avgFrameTime = this.getAverageFrameTime();
    
    // During interaction, try to maintain higher frame rate
    if (isInteracting) {
      this.currentUpdateInterval = INTERACTION_UPDATE_INTERVAL;
      return this.currentUpdateInterval;
    }

    // Adjust update interval based on performance
    if (avgFrameTime > TARGET_FRAME_TIME * 1.5) {
      // Performance is poor, reduce update rate
      this.currentUpdateInterval = Math.min(
        this.currentUpdateInterval * 1.2,
        MAX_UPDATE_INTERVAL
      );
    } else if (avgFrameTime < TARGET_FRAME_TIME * 0.8) {
      // Performance is good, increase update rate
      this.currentUpdateInterval = Math.max(
        this.currentUpdateInterval * 0.8,
        MIN_UPDATE_INTERVAL
      );
    }

    return this.currentUpdateInterval;
  }

  reset() {
    this.frameTimes = [];
    this.lastFrameTime = 0;
    this.currentUpdateInterval = MIN_UPDATE_INTERVAL;
  }
}

// Rate limiting helper with adaptive interval
class UpdateThrottler {
  private lastUpdateTime: number = 0;
  private pendingUpdate: ArrayBuffer | null = null;
  private timeoutId: number | null = null;
  private performanceMonitor: PerformanceMonitor;

  constructor() {
    this.performanceMonitor = new PerformanceMonitor();
  }

  addUpdate(data: ArrayBuffer, isInteracting: boolean): void {
    this.pendingUpdate = data;
    
    if (this.timeoutId === null) {
      const now = performance.now();
      this.performanceMonitor.addFrameTime(now);
      
      const updateInterval = this.performanceMonitor.calculateUpdateInterval(isInteracting);
      const timeSinceLastUpdate = now - this.lastUpdateTime;
      
      if (timeSinceLastUpdate >= updateInterval) {
        this.processPendingUpdate();
      } else {
        this.timeoutId = window.setTimeout(
          () => this.processPendingUpdate(),
          updateInterval - timeSinceLastUpdate
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

  reset(): void {
    this.lastUpdateTime = 0;
    this.pendingUpdate = null;
    if (this.timeoutId !== null) {
      window.clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    this.performanceMonitor.reset();
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
    private interactionMode: 'server' | 'local' = 'server';
    private interactedNodes: Set<string> = new Set();

    constructor(config: Partial<WebSocketConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        const hostname = window.location.host;
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

    public startInteractionMode() {
        logger.log('websocket', 'Starting interaction mode');
        this.interactionMode = 'local';
        this.interactedNodes.clear();
    }

    public endInteractionMode() {
        logger.log('websocket', 'Ending interaction mode');
        this.interactionMode = 'server';
        this.interactedNodes.clear();
    }

    public addInteractedNode(nodeId: string) {
        if (this.interactedNodes.size < 2) {
            logger.log('websocket', `Adding interacted node: ${nodeId}`);
            this.interactedNodes.add(nodeId);
        }
    }

    public isNodeInteracted(nodeId: string): boolean {
        return this.interactedNodes.has(nodeId);
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

    private stopHeartbeat(): void {
        if (this.heartbeatInterval) {
            window.clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
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
        // Only process binary updates if we're in server mode or if the data is for non-interacted nodes
        if (this.interactionMode === 'server' || !this.pendingBinaryUpdate) {
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

        // Only send binary updates for interacted nodes in local mode
        if (this.interactionMode === 'local' && this.interactedNodes.size > 0) {
            // Validate data size
            const expectedSize = this.indexToNodeId.length * BINARY_UPDATE_NODE_SIZE;
            if (data.byteLength !== expectedSize) {
                logger.log('binary', 'Invalid binary data size', {
                    expected: expectedSize,
                    received: data.byteLength
                });
                return;
            }

            // Add to throttled updates with interaction state
            this.updateThrottler.addUpdate(data, this.interactionMode === 'local');
        }
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
            this.interactedNodes.clear();
            this.interactionMode = 'server';
            this.updateThrottler.reset();
        }
    }
}
