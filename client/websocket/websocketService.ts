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
import { WS_MESSAGE_QUEUE_SIZE } from '../core/constants';
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
  private isConnecting = false;
  private reconnectAttempts = 0;
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

  constructor() {
    // Default settings, will be updated from settingsManager
    this.settings = {
      compressionEnabled: true,
      compressionThreshold: 1024,
      heartbeatInterval: 30000, // 30 seconds - match server's HEARTBEAT_INTERVAL
      heartbeatTimeout: 60000,  // 60 seconds - match server's CLIENT_TIMEOUT
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

    // Update WebSocket settings from server settings
    if (settings.websocket) {
      this.settings = {
        ...this.settings,
        compressionEnabled: settings.websocket.compressionEnabled,
        compressionThreshold: settings.websocket.compressionThreshold,
        heartbeatInterval: settings.websocket.heartbeatInterval * 1000, // Convert to ms
        heartbeatTimeout: settings.websocket.heartbeatTimeout * 1000,   // Convert to ms
        reconnectAttempts: settings.websocket.reconnectAttempts,
        reconnectDelay: settings.websocket.reconnectDelay
      };
    }

    this.connect();
  }

  private startHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    // Reset last pong time when starting heartbeat
    this.lastPongTime = Date.now();

    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected) {
        // Check if we haven't received a pong in too long
        const timeSinceLastPong = Date.now() - this.lastPongTime;
        if (timeSinceLastPong > this.settings.heartbeatTimeout) {
          logger.warn('WebSocket heartbeat timeout', {
            timeSinceLastPong,
            heartbeatTimeout: this.settings.heartbeatTimeout
          });
          this.handleConnectionFailure(new WebSocketError(
            CoreWebSocketErrorType.TIMEOUT,
            new Error('Heartbeat timeout'),
            1001,
            { timeSinceLastPong }
          ));
          return;
        }

        // Send ping with timestamp
        const pingMessage = {
          type: 'ping',
          timestamp: Date.now()
        };

        try {
          this.send(JSON.stringify(pingMessage));
          logger.debug('Sent ping', { timestamp: pingMessage.timestamp });
        } catch (error) {
          logger.error('Failed to send ping:', error);
        }
      }
    }, this.settings.heartbeatInterval);

    logger.debug('Started heartbeat timer', {
      interval: this.settings.heartbeatInterval,
      timeout: this.settings.heartbeatTimeout
    });
  }

  private connect(): void {
    try {
      // Get WebSocket URL from settings
      const wsUrl = settingsManager.getCurrentSettings().websocket?.url || '/wss';
      const url = new URL(wsUrl, window.location.href);
      url.protocol = url.protocol.replace('http', 'ws');
      const fullUrl = url.toString();

      logger.info('Connecting to WebSocket:', { url: fullUrl });

      // Create WebSocket connection
      logger.info('Creating WebSocket connection to:', { url: fullUrl });
      this.ws = new WebSocket(fullUrl);
      this.isConnected = false;
      this.isConnecting = true;

      // Log initial state
      if (this.ws) {
        logger.debug('WebSocket initial state:', {
          readyState: this.ws.readyState,
          connecting: this.ws.readyState === WebSocket.CONNECTING,
          protocol: this.ws.protocol,
          extensions: this.ws.extensions,
          timestamp: new Date().toISOString(),
          binaryType: this.ws.binaryType,
          bufferedAmount: this.ws.bufferedAmount
        });
      }

      // Setup event handlers
      this.ws.onopen = () => {
        logger.info('WebSocket connected:', {
          url: fullUrl,
          protocol: this.ws?.protocol,
          extensions: this.ws?.extensions,
          binaryType: this.ws?.binaryType
        });
        this.isConnected = true;
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        
        // Request binary updates immediately after connection
        this.send(JSON.stringify({
          type: 'enableBinaryUpdates',
          enabled: true
        }));
        
        this.startHeartbeat();
        this.notifyConnectionHandlers();

        // Log connection details
        if (this.ws) {
          logger.debug('Connection established:', {
            readyState: this.ws.readyState,
            protocol: this.ws.protocol,
            extensions: this.ws.extensions,
            timestamp: new Date().toISOString()
          });
        }
      };

      this.ws.onclose = (event) => {
        const wasConnected = this.isConnected;
        this.isConnected = false;
        this.isConnecting = false;
        
        if (this.heartbeatTimer) {
          clearInterval(this.heartbeatTimer);
          this.heartbeatTimer = null;
        }

        // Log close details
        logger.debug('WebSocket closed:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
          wasConnected,
          attempts: this.reconnectAttempts,
          timestamp: new Date().toISOString()
        });

        // Only attempt reconnect if we were previously connected
        // or if we're still within our reconnect attempts
        if (wasConnected || this.reconnectAttempts < this.settings.reconnectAttempts) {
          logger.warn('WebSocket closed, attempting reconnect', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean,
            attempt: this.reconnectAttempts + 1,
            maxAttempts: this.settings.reconnectAttempts
          });
          
          setTimeout(() => {
            if (!this.isConnected && !this.isConnecting) {
              this.reconnectAttempts++;
              this.connect();
            }
          }, this.settings.reconnectDelay);
        } else {
          logger.error('WebSocket connection failed', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean,
            attempts: this.reconnectAttempts
          });
          
          this.handleConnectionFailure(new WebSocketError(
            CoreWebSocketErrorType.CONNECTION_LOST,
            new Error(event.reason || 'Connection closed'),
            event.code
          ));
        }
      };

      this.ws.onerror = (event) => {
        logger.error('WebSocket error:', {
          event,
          readyState: this.ws?.readyState,
          timestamp: new Date().toISOString()
        });
        // Don't handle error here, let onclose handle it
        // This prevents duplicate error handling
      };

      this.ws.onmessage = (event) => {
        const settings = settingsManager.getCurrentSettings();
        const debugEnabled = settings.clientDebug.enabled;
        const websocketDebug = settings.clientDebug.enableWebsocketDebug;

        if (debugEnabled && websocketDebug) {
          logger.debug('Received message:', {
            type: typeof event.data,
            isBinary: event.data instanceof ArrayBuffer,
            size: event.data.length,
            readyState: this.ws?.readyState,
            timestamp: new Date().toISOString()
          });
        }

        try {
          if (typeof event.data === 'string') {
            const message = JSON.parse(event.data);
            this.handleJsonMessage(message);
          } else if (event.data instanceof ArrayBuffer) {
            this.handleBinaryMessage(event.data);
          } else {
            throw new Error('Unsupported message format');
          }
        } catch (error) {
          logger.error('Error handling message:', error);
          this.notifyErrorHandlers(new WebSocketError(
            CoreWebSocketErrorType.MESSAGE_PARSE_ERROR,
            error instanceof Error ? error : new Error('Failed to parse message'),
            1007
          ));
        }
      };

    } catch (error) {
      logger.error('Failed to create WebSocket:', error);
      this.handleConnectionFailure(new WebSocketError(
        CoreWebSocketErrorType.CONNECTION_ERROR,
        error instanceof Error ? error : new Error('Failed to create WebSocket'),
        1006
      ));
    }
  }

  private handleBinaryMessage(data: ArrayBuffer): void {
    const settings = settingsManager.getCurrentSettings();
    const debugEnabled = settings.clientDebug.enabled;
    const websocketDebug = settings.clientDebug.enableWebsocketDebug;

    try {
      // First 4 bytes are the header containing version
      const headerView = new DataView(data, 0, 4);
      const version = headerView.getInt32(0, true); // true for little-endian

      if (debugEnabled && websocketDebug) {
        logger.debug('Handling binary message:', {
          size: data.byteLength,
          version,
          readyState: this.ws?.readyState,
          timestamp: new Date().toISOString(),
          floatCount: (data.byteLength - 4) / 4, // Each float is 4 bytes
          expectedFloats: Math.floor((data.byteLength - 4) / 24) * 6 // 6 floats per node
        });
      }

      // Log the first few floats for debugging
      if (debugEnabled && websocketDebug && settings.clientDebug.logBinaryHeaders) {
        const floatArray = new Float32Array(data, 4); // Skip 4-byte header
        logger.debug('Binary data preview:', {
          firstNode: {
            position: {
              x: floatArray[0],
              y: floatArray[1],
              z: floatArray[2]
            },
            velocity: {
              x: floatArray[3],
              y: floatArray[4],
              z: floatArray[5]
            }
          }
        });
      }

      // Process binary data
      const floatArray = new Float32Array(data, 4); // Skip 4-byte version header
      const nodeCount = Math.floor(floatArray.length / 6); // 6 floats per node (pos + vel)
      
      // Notify handlers with binary position update
      const handlers = this.messageHandlers.get('binaryPositionUpdate' as MessageType);
      if (handlers) {
        const nodes = Array(nodeCount).fill(null).map((_, i) => ({
          data: {
            position: {
              x: floatArray[i * 6],
              y: floatArray[i * 6 + 1],
              z: floatArray[i * 6 + 2]
            },
            velocity: {
              x: floatArray[i * 6 + 3],
              y: floatArray[i * 6 + 4],
              z: floatArray[i * 6 + 5]
            }
          }
        }));

        handlers.forEach(handler => {
          try {
            handler({ type: 'binaryPositionUpdate', data: { nodes } });
          } catch (error) {
            logger.error('Error in binary message handler:', error);
          }
        });
      }
    } catch (error) {
      logger.error('Error parsing binary message:', {
        error,
        size: data.byteLength,
        readyState: this.ws?.readyState,
        timestamp: new Date().toISOString()
      });
      this.notifyErrorHandlers(new WebSocketError(
        CoreWebSocketErrorType.MESSAGE_PARSE_ERROR,
        error instanceof Error ? error : new Error('Failed to parse binary message'),
        1007
      ));
    }
  }

  private handleConnectionFailure(error: WebSocketError): void {
    const settings = settingsManager.getCurrentSettings();
    const debugEnabled = settings.clientDebug.enabled;
    const websocketDebug = settings.clientDebug.enableWebsocketDebug;

    if (debugEnabled && websocketDebug) {
      logger.warn('Handling connection failure:', {
        reconnectCount: this.reconnectAttempts,
        maxAttempts: this.settings.reconnectAttempts,
        error: error.message,
        readyState: this.ws?.readyState,
        timestamp: new Date().toISOString()
      });
    }

    this.notifyErrorHandlers(error);
  }

  private handleJsonMessage(message: any): void {
    const settings = settingsManager.getCurrentSettings();
    const debugEnabled = settings.clientDebug.enabled;
    const websocketDebug = settings.clientDebug.enableWebsocketDebug;

    if (debugEnabled && websocketDebug) {
      logger.debug('Handling JSON message:', {
        type: message?.type,
        readyState: this.ws?.readyState,
        timestamp: new Date().toISOString()
      });
    }

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
      const latency = this.lastPongTime - processedMessage.timestamp;
      if (debugEnabled && websocketDebug) {
        logger.debug('Received pong:', {
          timestamp: processedMessage.timestamp,
          latency,
          lastPongTime: this.lastPongTime
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

  private notifyConnectionHandlers(): void {
    this.connectionHandlers.forEach(handler => {
      try {
        handler(WebSocketStatus.CONNECTED);
      } catch (error) {
        logger.error('Error in connection handler:', error);
      }
    });
  }

  public send(data: string): void {
    const settings = settingsManager.getCurrentSettings();
    const debugEnabled = settings.clientDebug.enabled;
    const websocketDebug = settings.clientDebug.enableWebsocketDebug;

    if (!this.isConnected) {
      if (this.messageQueue.length < WS_MESSAGE_QUEUE_SIZE) {
        if (debugEnabled && websocketDebug) {
          logger.debug('Queuing message:', {
            queueLength: this.messageQueue.length,
            readyState: this.ws?.readyState,
            timestamp: new Date().toISOString()
          });
        }
        this.messageQueue.push(data);
      } else {
        logger.warn('Message queue full, dropping message');
      }
      return;
    }

    try {
      if (debugEnabled && websocketDebug) {
        logger.debug('Sending message:', {
          size: data.length,
          readyState: this.ws?.readyState,
          timestamp: new Date().toISOString()
        });
      }
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
