/**
 * WebSocket service for real-time communication
 */

import { WebSocketMessage, MessageType } from '../core/types';
import { WS_RECONNECT_INTERVAL, WS_HEARTBEAT_INTERVAL, WS_MESSAGE_QUEUE_SIZE } from '../core/constants';
import { createLogger } from '../core/utils';

const logger = createLogger('WebSocketService');

type MessageHandler = (data: any) => void;
type ErrorHandler = (error: Error) => void;

export class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectTimeout: number | null = null;
  private heartbeatInterval: number | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private messageHandlers: Map<MessageType, MessageHandler[]> = new Map();
  private errorHandlers: ErrorHandler[] = [];
  private isConnected: boolean = false;

  constructor(url: string) {
    this.url = url;
    this.initializeHandlers();
  }

  private initializeHandlers(): void {
    // Initialize handlers for each message type
    const messageTypes: MessageType[] = [
      'initialData',
      'graphUpdate',
      'binaryPositionUpdate',
      'settingsUpdate',
      'error'
    ];
    messageTypes.forEach(type => this.messageHandlers.set(type, []));
  }

  connect(): void {
    if (this.ws) {
      this.ws.close();
    }

    try {
      this.ws = new WebSocket(this.url);
      this.setupWebSocket();
    } catch (error) {
      logger.error('Failed to create WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }

  private setupWebSocket(): void {
    if (!this.ws) return;

    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      logger.log('WebSocket connected');
      this.isConnected = true;
      this.startHeartbeat();
      this.processMessageQueue();
    };

    this.ws.onclose = () => {
      logger.warn('WebSocket disconnected');
      this.isConnected = false;
      this.cleanup();
      this.scheduleReconnect();
    };

    this.ws.onerror = (event) => {
      logger.error('WebSocket error:', event);
      this.notifyErrorHandlers(new Error('WebSocket error occurred'));
    };

    this.ws.onmessage = (event) => {
      this.handleMessage(event);
    };
  }

  private handleMessage(event: MessageEvent): void {
    try {
      if (event.data instanceof ArrayBuffer) {
        // Handle binary message (position updates)
        this.notifyHandlers('binaryPositionUpdate', event.data);
      } else {
        // Handle JSON message
        const message = JSON.parse(event.data) as WebSocketMessage;
        this.notifyHandlers(message.type, message.data);
      }
    } catch (error) {
      logger.error('Error handling message:', error);
      this.notifyErrorHandlers(new Error('Failed to process message'));
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
    this.errorHandlers.forEach(handler => {
      try {
        handler(error);
      } catch (error) {
        logger.error('Error in error handler:', error);
      }
    });
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatInterval = window.setInterval(() => {
      if (this.isConnected) {
        this.send({ type: 'heartbeat', data: null });
      }
    }, WS_HEARTBEAT_INTERVAL);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval !== null) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout === null) {
      this.reconnectTimeout = window.setTimeout(() => {
        this.reconnectTimeout = null;
        this.connect();
      }, WS_RECONNECT_INTERVAL);
    }
  }

  private cleanup(): void {
    this.stopHeartbeat();
    if (this.reconnectTimeout !== null) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  private processMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }
  }

  // Public API

  send(message: WebSocketMessage): void {
    if (!this.isConnected) {
      if (this.messageQueue.length < WS_MESSAGE_QUEUE_SIZE) {
        this.messageQueue.push(message);
      } else {
        logger.warn('Message queue full, dropping message');
      }
      return;
    }

    try {
      this.ws?.send(JSON.stringify(message));
    } catch (error) {
      logger.error('Error sending message:', error);
      this.notifyErrorHandlers(new Error('Failed to send message'));
    }
  }

  sendBinary(data: ArrayBuffer): void {
    if (!this.isConnected) {
      logger.warn('Cannot send binary data while disconnected');
      return;
    }

    try {
      this.ws?.send(data);
    } catch (error) {
      logger.error('Error sending binary data:', error);
      this.notifyErrorHandlers(new Error('Failed to send binary data'));
    }
  }

  on(type: MessageType, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      handlers.push(handler);
    }
  }

  onError(handler: ErrorHandler): void {
    this.errorHandlers.push(handler);
  }

  off(type: MessageType, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      const __index = handlers.indexOf(handler);
      if (__index !== -1) {
        handlers.splice(__index, 1);
      }
    }
  }

  offError(handler: ErrorHandler): void {
    const __index = this.errorHandlers.indexOf(handler);
    if (__index !== -1) {
      this.errorHandlers.splice(__index, 1);
    }
  }

  disconnect(): void {
    this.cleanup();
    this.ws?.close();
  }

  isConnectedToServer(): boolean {
    return this.isConnected;
  }
}
