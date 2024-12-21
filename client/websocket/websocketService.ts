/**
 * WebSocket service for real-time communication
 */

import {
  MessageType,
  MessageHandler,
  ErrorHandler,
  ConnectionHandler,
  WebSocketStatus,
  WebSocketError as CoreWebSocketError,
  WebSocketErrorType as CoreWebSocketErrorType
} from '../core/types';
import { WS_MESSAGE_QUEUE_SIZE, WS_URL, BINARY_VERSION } from '../core/constants';
import { createLogger } from '../utils/logger';
import { settingsManager } from '../state/settings';

const logger = createLogger('WebSocketService');

// WebSocket error class
export class WebSocketError extends Error implements CoreWebSocketError {
  constructor(
    public readonly type: CoreWebSocketErrorType,
    public readonly originalError?: Error,
    public readonly code?: number,
    public readonly details?: any
  ) {
    super(originalError?.message || type);
    this.name = 'WebSocketError';
  }
}

export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private messageQueue: Array<string> = [];
  private isConnected = false;
  private messageHandlers: Map<MessageType, MessageHandler[]> = new Map();
  private errorHandlers: ErrorHandler[] = [];
  private connectionHandlers: ConnectionHandler[] = [];
  private settings: {
    compressionEnabled: boolean;
    compressionThreshold: number;
    heartbeatInterval: number;
    heartbeatTimeout: number;
    reconnectAttempts: number;
    reconnectDelay: number;
  };
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private lastPongTime: number = 0;
  private reconnectCount: number = 0;

  constructor() {
    // Default settings, will be updated from settingsManager
    this.settings = {
      compressionEnabled: true,
      compressionThreshold: 1024,
      heartbeatInterval: 15000,
      heartbeatTimeout: 60000,
      reconnectAttempts: 3,
      reconnectDelay: 5000
    };

    // Get debug settings from settings manager
    const settings = settingsManager.getCurrentSettings();
    const debugEnabled = settings.clientDebug.enabled;
    const websocketDebug = settings.clientDebug.enableWebsocketDebug;
    
    if (debugEnabled && websocketDebug) {
      logger.info('WebSocket debug logging enabled');
    }

    this.connect();
  }

  private startHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected) {
        // Check if we haven't received a pong in too long
        const timeSinceLastPong = Date.now() - this.lastPongTime;
        if (timeSinceLastPong > this.settings.heartbeatTimeout) {
          logger.warn('WebSocket heartbeat timeout');
          this.handleConnectionFailure(new WebSocketError(
            CoreWebSocketErrorType.TIMEOUT,
            new Error('Heartbeat timeout'),
            1001,
            { timeSinceLastPong }
          ));
          return;
        }

        // Send ping
        this.send(JSON.stringify({
          type: 'ping',
          timestamp: Date.now()
        }));
      }
    }, this.settings.heartbeatInterval);
  }

  private connect(): void {
    try {
      const settings = settingsManager.getCurrentSettings();
      const debugEnabled = settings.clientDebug.enabled;
      const websocketDebug = settings.clientDebug.enableWebsocketDebug;

      if (debugEnabled && websocketDebug) {
        logger.info('Attempting WebSocket connection to:', WS_URL);
        logger.debug('Current WebSocket state:', {
          isConnected: this.isConnected,
          reconnectCount: this.reconnectCount,
          queueSize: this.messageQueue.length
        });
      }

      this.ws = new WebSocket(WS_URL);
      this.setupEventHandlers();
      this.startHeartbeat();
    } catch (error) {
      logger.error('WebSocket connection error:', error);
      this.handleConnectionFailure(new WebSocketError(
        CoreWebSocketErrorType.CONNECTION_FAILED,
        error instanceof Error ? error : new Error('Failed to create WebSocket'),
        1006
      ));
    }
  }

  private setupEventHandlers(): void {
    if (!this.ws) {
      logger.error('Cannot setup event handlers: WebSocket is null');
      return;
    }

    const settings = settingsManager.getCurrentSettings();
    const debugEnabled = settings.clientDebug.enabled;
    const websocketDebug = settings.clientDebug.enableWebsocketDebug;

    this.ws.onopen = () => {
      if (debugEnabled && websocketDebug) {
        logger.info('WebSocket connection established successfully');
        logger.debug('Connection details:', {
          url: WS_URL,
          protocol: this.ws?.protocol,
          readyState: this.ws?.readyState,
          extensions: this.ws?.extensions
        });
      }

      this.isConnected = true;
      this.reconnectCount = 0;
      this.lastPongTime = Date.now();
      this.notifyConnectionHandlers(WebSocketStatus.CONNECTED);

      // Send initial handshake message
      this.send(JSON.stringify({
        type: 'handshake',
        version: BINARY_VERSION,
        timestamp: Date.now()
      }));

      this.flushMessageQueue();
    };

    this.ws.onclose = (event) => {
      if (debugEnabled && websocketDebug) {
        logger.warn('WebSocket connection closed:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
          timestamp: Date.now()
        });
      }

      this.isConnected = false;
      this.notifyConnectionHandlers(WebSocketStatus.DISCONNECTED, {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean
      });
      this.handleConnectionFailure(new WebSocketError(
        CoreWebSocketErrorType.CONNECTION_LOST,
        new Error(event.reason || 'Connection closed'),
        event.code
      ));
    };

    this.ws.onerror = (error) => {
      if (debugEnabled && websocketDebug) {
        logger.error('WebSocket error:', error);
        logger.debug('WebSocket state at error:', {
          readyState: this.ws?.readyState,
          isConnected: this.isConnected,
          reconnectCount: this.reconnectCount,
          lastPongTime: this.lastPongTime
        });
      }

      this.notifyErrorHandlers(new WebSocketError(
        CoreWebSocketErrorType.CONNECTION_FAILED,
        error instanceof Error ? error : new Error('Unknown WebSocket error'),
        1006
      ));
    };

    this.ws.onmessage = (event) => {
      if (debugEnabled && websocketDebug) {
        logger.debug('WebSocket message received:', {
          type: 'text',
          size: event.data.length
        });
      }

      try {
        const message = JSON.parse(event.data);
        if (debugEnabled && websocketDebug) {
          logger.debug('Parsed WebSocket message:', message);
        }
        this.handleJsonMessage(message);
      } catch (error) {
        logger.error('Failed to parse WebSocket message:', error);
        this.notifyErrorHandlers(new WebSocketError(
          CoreWebSocketErrorType.MESSAGE_PARSE_ERROR,
          error instanceof Error ? error : new Error('Failed to parse message'),
          1007
        ));
      }
    };
  }

  private handleConnectionFailure(error: WebSocketError): void {
    const settings = settingsManager.getCurrentSettings();
    const debugEnabled = settings.clientDebug.enabled;
    const websocketDebug = settings.clientDebug.enableWebsocketDebug;

    if (debugEnabled && websocketDebug) {
      logger.warn('Handling connection failure:', {
        reconnectCount: this.reconnectCount,
        maxAttempts: this.settings.reconnectAttempts,
        error: error.message
      });
    }

    this.reconnectCount++;
    if (this.reconnectCount > this.settings.reconnectAttempts) {
      logger.error('Max reconnection attempts reached');
      this.notifyErrorHandlers(new WebSocketError(
        CoreWebSocketErrorType.MAX_RETRIES_EXCEEDED,
        new Error('Maximum reconnection attempts reached'),
        1011,
        { attempts: this.reconnectCount }
      ));
      this.notifyConnectionHandlers(WebSocketStatus.FAILED);
      return;
    }

    this.notifyConnectionHandlers(WebSocketStatus.RECONNECTING);

    if (debugEnabled && websocketDebug) {
      logger.info(`Reconnection attempt ${this.reconnectCount}/${this.settings.reconnectAttempts}`);
    }
    this.scheduleReconnect();
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, this.settings.reconnectDelay);
  }

  private handleJsonMessage(message: any): void {
    const settings = settingsManager.getCurrentSettings();
    const debugEnabled = settings.clientDebug.enabled;
    const websocketDebug = settings.clientDebug.enableWebsocketDebug;

    if (!message || !message.type) {
      if (debugEnabled && websocketDebug) {
        logger.warn('Invalid message format:', message);
      }
      this.notifyErrorHandlers(new WebSocketError(
        CoreWebSocketErrorType.INVALID_MESSAGE,
        new Error('Invalid message format: missing type'),
        1003
      ));
      return;
    }

    // Convert snake_case to camelCase if needed
    const processedMessage = typeof message === 'object' ? 
      Object.keys(message).reduce((acc: any, key: string) => {
        acc[key.replace(/_([a-z])/g, g => g[1].toUpperCase())] = message[key];
        return acc;
      }, {}) : message;

    // Handle pong messages for heartbeat
    if (processedMessage.type === 'pong') {
      this.lastPongTime = Date.now();
      if (debugEnabled && websocketDebug) {
        logger.debug('Received pong:', {
          timestamp: processedMessage.timestamp,
          latency: Date.now() - processedMessage.timestamp
        });
      }
      return;
    }

    // Notify message handlers
    const handlers = this.messageHandlers.get(processedMessage.type as MessageType);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(processedMessage);
        } catch (error) {
          logger.error('Error in message handler:', error);
          this.notifyErrorHandlers(new WebSocketError(
            CoreWebSocketErrorType.MESSAGE_PARSE_ERROR,
            error instanceof Error ? error : new Error('Error in message handler'),
            1011
          ));
        }
      });
    }
  }

  private notifyErrorHandlers(error: CoreWebSocketError): void {
    this.errorHandlers.forEach(handler => {
      try {
        handler(error);
      } catch (handlerError) {
        logger.error('Error in error handler:', handlerError);
      }
    });
  }

  private notifyConnectionHandlers(status: WebSocketStatus, details?: any): void {
    this.connectionHandlers.forEach(handler => {
      try {
        handler(status, details);
      } catch (error) {
        logger.error('Error in connection handler:', error);
      }
    });
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }

  public send(data: string): void {
    if (!this.isConnected) {
      if (this.messageQueue.length < WS_MESSAGE_QUEUE_SIZE) {
        this.messageQueue.push(data);
      } else {
        logger.warn('Message queue full, dropping message');
      }
      return;
    }

    try {
      this.ws?.send(data);
    } catch (error) {
      logger.error('Error sending message:', error);
      this.notifyErrorHandlers(new WebSocketError(
        CoreWebSocketErrorType.SEND_FAILED,
        error instanceof Error ? error : new Error('Failed to send message'),
        1011
      ));
      this.handleConnectionFailure(new WebSocketError(
        CoreWebSocketErrorType.CONNECTION_LOST,
        error instanceof Error ? error : new Error('Connection lost after send failure'),
        1006
      ));
    }
  }

  // Alias for backward compatibility
  public on = this.onMessage;

  public onMessage(type: MessageType, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(type) || [];
    handlers.push(handler);
    this.messageHandlers.set(type, handlers);
  }

  public onError(handler: ErrorHandler): void {
    this.errorHandlers.push(handler);
  }

  public onConnectionChange(handler: ConnectionHandler): void {
    this.connectionHandlers.push(handler);
    handler(
      this.isConnected ? WebSocketStatus.CONNECTED : WebSocketStatus.DISCONNECTED,
      undefined
    );
  }

  // Alias for backward compatibility
  public dispose = this.disconnect;

  public disconnect(): void {
    if (this.ws) {
      this.ws.close();
    }
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
  }
}
