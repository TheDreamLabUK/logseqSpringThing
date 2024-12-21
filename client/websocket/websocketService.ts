/**
 * WebSocket service for real-time communication
 */

import { 
  MessageType,
  MessageHandler,
  ErrorHandler,
  ConnectionHandler,
  WebSocketErrorType,
  WebSocketError,
  WebSocketStatus
} from '../core/types';
import { WS_RECONNECT_INTERVAL, WS_MESSAGE_QUEUE_SIZE, WS_URL, BINARY_VERSION } from '../core/constants';
import { createLogger } from '../core/utils';
import { convertObjectKeysToCamelCase } from '../core/utils';
import { settingsManager } from '../state/settings';

const logger = createLogger('WebSocketService');

// Constants
const BINARY_HEADER_SIZE = 4; // 1 float * 4 bytes

export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private messageQueue: Array<ArrayBuffer | string> = [];
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

  private connect(): void {
    try {
      // Note: Browser WebSocket API doesn't support compression options directly
      // Compression is handled by the server configuration
      this.ws = new WebSocket(WS_URL);
      this.setupEventHandlers();
      this.startHeartbeat();
    } catch (error) {
      logger.error('WebSocket connection error:', error);
      this.handleConnectionFailure();
    }
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
          this.handleConnectionFailure();
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

  private handleConnectionFailure(): void {
    this.reconnectCount++;
    if (this.reconnectCount > this.settings.reconnectAttempts) {
      logger.error('Max reconnection attempts reached');
      this.notifyErrorHandlers(new Error('Max reconnection attempts reached'));
      this.notifyConnectionHandlers(WebSocketStatus.FAILED);
      return;
    }

    logger.info(`Reconnection attempt ${this.reconnectCount}/${this.settings.reconnectAttempts}`);
    this.scheduleReconnect();
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      this.isConnected = true;
      this.reconnectCount = 0;
      this.lastPongTime = Date.now();
      const settings = settingsManager.getCurrentSettings();
      const debugEnabled = settings.clientDebug.enabled;
      const websocketDebug = settings.clientDebug.enableWebsocketDebug;
      if (debugEnabled && websocketDebug) {
        logger.info('WebSocket connected');
      }

      this.flushMessageQueue();
      this.notifyConnectionHandlers(WebSocketStatus.CONNECTED);
    };

    this.ws.onclose = (event) => {
      this.isConnected = false;
      logger.info(`WebSocket disconnected: ${event.code} - ${event.reason}`);
      const settings = settingsManager.getCurrentSettings();
      const debugEnabled = settings.clientDebug.enabled;
      const websocketDebug = settings.clientDebug.enableWebsocketDebug;
      if (debugEnabled && websocketDebug) {
        logger.debug('WebSocket close details:', {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
          timestamp: new Date().toISOString()
        });
      }

      this.handleConnectionFailure();
      this.notifyConnectionHandlers(WebSocketStatus.DISCONNECTED, {
        code: event.code,
        reason: event.reason
      });
    };

    this.ws.onerror = (error) => {
      logger.error('WebSocket error:', error);
      const settings = settingsManager.getCurrentSettings();
      const debugEnabled = settings.clientDebug.enabled;
      const websocketDebug = settings.clientDebug.enableWebsocketDebug;
      if (debugEnabled && websocketDebug) {
        logger.debug('WebSocket error details:', {
          error: error instanceof Error ? {
            name: error.name,
            message: error.message,
            stack: error.stack
          } : error,
          readyState: this.ws?.readyState,
          timestamp: new Date().toISOString()
        });
      }

      const wsError: WebSocketError = {
        name: 'WebSocketError',
        message: error instanceof Error ? error.message : 'Unknown WebSocket error',
        type: WebSocketErrorType.CONNECTION_FAILED,
        code: (error as any).code
      };
      this.notifyErrorHandlers(wsError);
    };

    this.ws.onmessage = (event) => {
      const settings = settingsManager.getCurrentSettings();
      const debugEnabled = settings.clientDebug.enabled;
      const websocketDebug = settings.clientDebug.enableWebsocketDebug;
      if (debugEnabled && websocketDebug) {
        const isBinary = event.data instanceof Blob;
        logger.debug('WebSocket message received:', {
          type: isBinary ? 'binary' : 'text',
          size: isBinary ? event.data.size : event.data.length,
          timestamp: new Date().toISOString()
        });
      }

      try {
        if (event.data instanceof Blob) {
          // Handle binary message
          event.data.arrayBuffer().then(buffer => {
            this.handleBinaryMessage(buffer);
          });
        } else {
          // Handle JSON message
          const message = JSON.parse(event.data);
          this.handleJsonMessage(message);
        }
      } catch (error) {
        logger.error('Error handling WebSocket message:', error);
      }
    };
  }

  private handleBinaryMessage(buffer: ArrayBuffer): void {
    try {
      const dataView = new DataView(buffer);
      const version = dataView.getFloat32(0, true); // true for little-endian
      
      if (version === BINARY_VERSION) {
        const nodeCount = (buffer.byteLength - BINARY_HEADER_SIZE) / (6 * 4); // 6 floats per node (pos + vel)
        const positions = new Float32Array(nodeCount * 3);
        const velocities = new Float32Array(nodeCount * 3);
        
        // Read position and velocity data directly using established node order
        for (let i = 0; i < nodeCount; i++) {
          const offset = BINARY_HEADER_SIZE + i * 6 * 4;
          
          // Read position
          positions[i * 3] = dataView.getFloat32(offset, true);      // x
          positions[i * 3 + 1] = dataView.getFloat32(offset + 4, true);  // y
          positions[i * 3 + 2] = dataView.getFloat32(offset + 8, true);  // z
          
          // Read velocity
          velocities[i * 3] = dataView.getFloat32(offset + 12, true);    // vx
          velocities[i * 3 + 1] = dataView.getFloat32(offset + 16, true); // vy
          velocities[i * 3 + 2] = dataView.getFloat32(offset + 20, true); // vz
        }
        
        this.notifyHandlers('binaryPositionUpdate', { 
          positions,
          velocities
        });
      } else {
        logger.warn(`Unsupported binary message version: ${version}`);
      }
    } catch (error) {
      logger.error('Error handling binary message:', error);
    }
  }

  private handleJsonMessage(message: any): void {
    const settings = settingsManager.getCurrentSettings();
    const debugEnabled = settings.clientDebug.enabled;
    const websocketDebug = settings.clientDebug.enableWebsocketDebug;
    const logFullJson = settings.clientDebug.logFullJson;
    if (debugEnabled && websocketDebug && logFullJson) {
      logger.debug('Handling JSON message:', message);
    }

    try {
      // Convert snake_case to camelCase
      const camelCaseMessage = convertObjectKeysToCamelCase(message);

      if (!camelCaseMessage.type) {
        logger.error('Invalid message format: missing type');
        return;
      }

      if (camelCaseMessage.type === 'ping') {
        this.send(JSON.stringify({ type: 'pong' }));
        return;
      }

      if (debugEnabled && websocketDebug) {
        logger.debug(`Processing message type: ${camelCaseMessage.type}`, {
          messageSize: JSON.stringify(message).length,
          hasData: !!camelCaseMessage.data,
          timestamp: new Date().toISOString()
        });
      }

      this.notifyHandlers(camelCaseMessage.type, camelCaseMessage.data);
    } catch (error) {
      logger.error('Error handling JSON message:', error);
      if (debugEnabled && websocketDebug) {
        logger.debug('Failed message:', message);
      }
    }
  }

  private notifyHandlers(type: MessageType, data: any): void {
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          logger.error(`Error in ${type} handler:`, error);
        }
      });
    }
  }

  private notifyErrorHandlers(error: Error): void {
    const wsError = {
      name: error.name,
      message: error.message,
      type: WebSocketErrorType.CONNECTION_FAILED,
      stack: error.stack
    } as WebSocketError;
    
    this.errorHandlers.forEach(handler => {
      try {
        handler(wsError);
      } catch (error) {
        logger.error('Error in error handler:', error);
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

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    
    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, WS_RECONNECT_INTERVAL);
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }

  public send(data: string | ArrayBuffer): void {
    if (this.isConnected && this.ws) {
      try {
        this.ws.send(data);
      } catch (error) {
        logger.error('Error sending WebSocket message:', error);
        this.messageQueue.push(data);
      }
    } else {
      this.messageQueue.push(data);
    }
    
    // Trim message queue if it gets too large
    if (this.messageQueue.length > WS_MESSAGE_QUEUE_SIZE) {
      this.messageQueue = this.messageQueue.slice(-WS_MESSAGE_QUEUE_SIZE);
    }
  }

  public dispose(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    // Clear all handlers
    this.messageHandlers.clear();
    this.errorHandlers = [];
    this.connectionHandlers = [];
  }

  public updateSettings(settings: Partial<typeof this.settings>): void {
    this.settings = { ...this.settings, ...settings };
    
    // Restart heartbeat with new settings if connected
    if (this.isConnected) {
      this.startHeartbeat();
    }
  }

  public on(type: MessageType, handler: MessageHandler): void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    const handlers = this.messageHandlers.get(type)!;
    handlers.push(handler);
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

  public off(type: MessageType, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index !== -1) {
        handlers.splice(index, 1);
      }
    }
  }

  public offError(handler: ErrorHandler): void {
    const index = this.errorHandlers.indexOf(handler);
    if (index !== -1) {
      this.errorHandlers.splice(index, 1);
    }
  }

  public offConnectionChange(handler: ConnectionHandler): void {
    const index = this.connectionHandlers.indexOf(handler);
    if (index !== -1) {
      this.connectionHandlers.splice(index, 1);
    }
  }
}
