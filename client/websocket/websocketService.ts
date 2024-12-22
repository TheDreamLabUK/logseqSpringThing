import {
  MessageType,
  WebSocketMessage,
  WebSocketErrorType,
  WebSocketSettings,
} from '../core/types';
import { createLogger } from '../utils/logger';

const logger = createLogger('WebSocketService');

const DEFAULT_SETTINGS: WebSocketSettings = {
  url: 'ws://localhost:8080/wss',  // Default WebSocket URL
  heartbeatInterval: 30,           // 30 seconds
  heartbeatTimeout: 60,            // 60 seconds
  reconnectAttempts: 3,           // 3 attempts
  reconnectDelay: 5000,           // 5 seconds
  binaryChunkSize: 65536,
  compressionEnabled: true,
  compressionThreshold: 1024,
  maxConnections: 1000,
  maxMessageSize: 100485760,
  updateRate: 90
};

interface Node {
  data: {
    position: { x: number; y: number; z: number };
    velocity: { x: number; y: number; z: number };
  };
}

type BinaryUpdateHandler = (data: ArrayBuffer) => void;
type MessageHandler = (data: any) => void;

export class WebSocketService {
  private ws: WebSocket | null = null;
  private settings: WebSocketSettings;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private lastPongTime: number = 0;
  private reconnectAttempts = 0;
  private binaryUpdateHandler: BinaryUpdateHandler | null = null;
  private isReconnecting = false;
  private messageHandlers: Map<MessageType, MessageHandler[]> = new Map();

  constructor(settings: Partial<WebSocketSettings> = {}) {
    this.settings = { ...DEFAULT_SETTINGS, ...settings };
    this.connect();
  }

  // Compatibility methods for existing code
  public onMessage(type: MessageType, handler: MessageHandler) {
    const handlers = this.messageHandlers.get(type) || [];
    handlers.push(handler);
    this.messageHandlers.set(type, handlers);
  }

  public send(data: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    }
  }

  // Alias for disconnect for backward compatibility
  public dispose() {
    this.disconnect();
  }

  public onBinaryUpdate(handler: BinaryUpdateHandler) {
    this.binaryUpdateHandler = handler;
  }

  private connect() {
    if (this.ws?.readyState === WebSocket.CONNECTING || this.isReconnecting) {
      return;
    }

    try {
      this.ws = new WebSocket(this.settings.url);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);

      this.startHeartbeat();
    } catch (error) {
      this.handleError(error as Event);
    }
  }

  private handleOpen() {
    logger.info('WebSocket connected');
    this.reconnectAttempts = 0;
    this.isReconnecting = false;
    this.lastPongTime = Date.now();
    
    // Notify connection status handlers
    const handlers = this.messageHandlers.get('connectionStatus' as MessageType);
    if (handlers) {
      handlers.forEach(h => h({ status: 'CONNECTED' }));
    }
  }

  private handleClose(event: CloseEvent) {
    this.cleanup();
    
    // Notify connection status handlers
    const handlers = this.messageHandlers.get('connectionStatus' as MessageType);
    if (handlers) {
      handlers.forEach(h => h({ status: 'DISCONNECTED', details: event }));
    }

    if (!event.wasClean) {
      this.attemptReconnect();
    }
  }

  private handleError(event: Event) {
    const error = {
      type: WebSocketErrorType.CONNECTION_ERROR,
      message: 'WebSocket connection error',
      details: event
    };
    logger.error('WebSocket error:', error);
    this.cleanup();
    this.attemptReconnect();
  }

  private handleMessage(event: MessageEvent) {
    this.lastPongTime = Date.now();

    if (event.data instanceof ArrayBuffer) {
      if (this.binaryUpdateHandler) {
        this.binaryUpdateHandler(event.data);
      }
      // Also notify through the traditional handler
      const handlers = this.messageHandlers.get('binaryPositionUpdate');
      if (handlers) {
        const view = new Float32Array(event.data);
        const nodes: Node[] = [];
        for (let i = 0; i < view.length; i += 6) {
          nodes.push({
            data: {
              position: { x: view[i], y: view[i+1], z: view[i+2] },
              velocity: { x: view[i+3], y: view[i+4], z: view[i+5] }
            }
          });
        }
        handlers.forEach(h => h({ type: 'binaryPositionUpdate', data: { nodes } }));
      }
      return;
    }

    try {
      const message = JSON.parse(event.data) as WebSocketMessage;
      
      // Handle message based on type
      const handlers = this.messageHandlers.get(message.type);
      if (handlers) {
        handlers.forEach(h => h(message));
      }

      switch (message.type) {
        case 'ping':
          this.handlePing(message);
          break;
        case 'pong':
          // Already updated lastPongTime
          break;
        default:
          logger.warn('Unhandled message type:', message.type);
      }
    } catch (error) {
      logger.error('Failed to parse message:', error);
    }
  }

  private handlePing(message: { type: 'ping', timestamp: number }) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const pong = {
        type: 'pong' as const,
        timestamp: message.timestamp
      };
      this.ws.send(JSON.stringify(pong));
    }
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState !== WebSocket.OPEN) {
        return;
      }

      // Check if we've received a pong recently
      const timeSinceLastPong = Date.now() - this.lastPongTime;
      if (timeSinceLastPong > this.settings.heartbeatTimeout * 1000) {
        logger.warn('Connection timeout, reconnecting...');
        this.cleanup();
        this.attemptReconnect();
        return;
      }

      // Send ping
      const ping = {
        type: 'ping' as const,
        timestamp: Date.now()
      };
      this.ws.send(JSON.stringify(ping));
    }, this.settings.heartbeatInterval * 1000);
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private cleanup() {
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.onmessage = null;
      
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.close();
      }
      this.ws = null;
    }
  }

  private attemptReconnect() {
    if (this.isReconnecting || this.reconnectAttempts >= this.settings.reconnectAttempts) {
      logger.error('Max reconnection attempts reached');
      return;
    }

    this.isReconnecting = true;
    this.reconnectAttempts++;

    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    this.reconnectTimeout = setTimeout(() => {
      logger.info(`Reconnecting... Attempt ${this.reconnectAttempts}`);
      this.connect();
    }, this.settings.reconnectDelay);
  }

  public disconnect() {
    this.cleanup();
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }
}
