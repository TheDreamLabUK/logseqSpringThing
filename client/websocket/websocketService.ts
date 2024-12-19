/**
 * WebSocket service for real-time communication
 */

import { 
  MessageType,
  MessageHandler,
  ErrorHandler,
  ConnectionHandler,
  PingMessage,
  WebSocketErrorType,
  WebSocketError,
  WebSocketStatus
} from '../core/types';
import { WS_RECONNECT_INTERVAL, WS_MESSAGE_QUEUE_SIZE, WS_URL, BINARY_VERSION } from '../core/constants';
import { createLogger } from '../core/utils';
import { convertObjectKeysToCamelCase } from '../core/utils';

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

  constructor() {
    this.connect();
  }

  private connect(): void {
    try {
      this.ws = new WebSocket(WS_URL);
      this.setupEventHandlers();
    } catch (error) {
      logger.error('WebSocket connection error:', error);
      this.scheduleReconnect();
    }
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      this.isConnected = true;
      logger.info('WebSocket connected');
      this.flushMessageQueue();
      this.notifyConnectionHandlers(WebSocketStatus.CONNECTED);
    };

    this.ws.onclose = () => {
      this.isConnected = false;
      logger.info('WebSocket disconnected');
      this.scheduleReconnect();
      this.notifyConnectionHandlers(WebSocketStatus.DISCONNECTED);
    };

    this.ws.onerror = (error) => {
      logger.error('WebSocket error:', error);
      this.notifyErrorHandlers(new Error('WebSocket error occurred'));
    };

    this.ws.onmessage = (event) => {
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
      const view = new Float32Array(buffer);
      const version = view[0];
      
      if (version === BINARY_VERSION) {
        // Extract node positions and velocities
        const positions = new Float32Array(buffer.slice(BINARY_HEADER_SIZE));
        this.notifyHandlers('binaryPositionUpdate', { positions });
      } else {
        logger.warn(`Unsupported binary message version: ${version}`);
      }
    } catch (error) {
      logger.error('Error handling binary message:', error);
    }
  }

  private handleJsonMessage(message: any): void {
    try {
      switch (message.type as MessageType) {
        case 'settingsUpdated':
          // Convert snake_case to camelCase and update settings
          const settings = convertObjectKeysToCamelCase(message.data);
          this.notifyHandlers('settingsUpdated', settings);
          break;
          
        case 'graphUpdated':
          this.notifyHandlers('graphUpdated', message.data);
          break;
          
        case 'ping':
          this.send(JSON.stringify({
            type: 'pong',
            timestamp: (message as PingMessage).timestamp
          }));
          break;
          
        default:
          if (this.messageHandlers.has(message.type)) {
            this.notifyHandlers(message.type, message.data);
          } else {
            logger.warn(`Unknown message type: ${message.type}`);
          }
      }
    } catch (error) {
      logger.error('Error handling JSON message:', error);
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

  private notifyConnectionHandlers(status: WebSocketStatus): void {
    this.connectionHandlers.forEach(handler => {
      try {
        handler(status);
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

    // Clear all handlers
    this.messageHandlers.clear();
    this.errorHandlers = [];
    this.connectionHandlers = [];
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
    handler(this.isConnected ? WebSocketStatus.CONNECTED : WebSocketStatus.DISCONNECTED);
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
