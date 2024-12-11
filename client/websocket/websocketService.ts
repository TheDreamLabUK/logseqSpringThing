/**
 * WebSocket service for real-time communication
 */

import { WebSocketMessage, MessageType } from '../core/types';
import { WS_RECONNECT_INTERVAL, WS_MESSAGE_QUEUE_SIZE } from '../core/constants';
import { createLogger } from '../core/utils';

const logger = createLogger('WebSocketService');

// Server heartbeat configuration from settings.toml
const HEARTBEAT_INTERVAL = 15000; // 15 seconds

type MessageHandler = (data: any) => void;
type ErrorHandler = (error: Error) => void;

// Network debug panel
class NetworkDebugPanel {
  private container: HTMLDivElement;
  private messageList: HTMLUListElement;
  private maxMessages = 50;

  constructor() {
    this.container = document.createElement('div');
    this.container.style.cssText = `
      position: fixed;
      top: 10px;
      left: 10px;
      background: rgba(0, 0, 0, 0.8);
      color: #00ff00;
      padding: 10px;
      border-radius: 5px;
      font-family: monospace;
      font-size: 12px;
      max-height: 300px;
      overflow-y: auto;
      z-index: 1000;
    `;

    const title = document.createElement('div');
    title.textContent = 'Network Messages';
    title.style.marginBottom = '5px';
    this.container.appendChild(title);

    this.messageList = document.createElement('ul');
    this.messageList.style.cssText = `
      list-style: none;
      margin: 0;
      padding: 0;
    `;
    this.container.appendChild(this.messageList);

    document.body.appendChild(this.container);
  }

  addMessage(direction: 'in' | 'out', message: any): void {
    const item = document.createElement('li');
    const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);
    const arrow = direction === 'in' ? '←' : '→';
    let displayMessage: string;

    if (message instanceof ArrayBuffer) {
      displayMessage = `Binary data (${message.byteLength} bytes)`;
    } else if (typeof message === 'string') {
      try {
        displayMessage = JSON.stringify(JSON.parse(message), null, 2);
      } catch {
        displayMessage = message;
      }
    } else {
      displayMessage = JSON.stringify(message, null, 2);
    }

    item.textContent = `${timestamp} ${arrow} ${displayMessage}`;
    item.style.marginBottom = '2px';
    item.style.wordBreak = 'break-all';
    item.style.whiteSpace = 'pre-wrap';

    this.messageList.insertBefore(item, this.messageList.firstChild);

    // Limit number of messages
    while (this.messageList.children.length > this.maxMessages) {
      this.messageList.removeChild(this.messageList.lastChild!);
    }
  }
}

export class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectTimeout: number | null = null;
  private heartbeatInterval: number | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private messageHandlers: Map<MessageType, MessageHandler[]> = new Map();
  private errorHandlers: ErrorHandler[] = [];
  private isConnected: boolean = false;
  private debugPanel: NetworkDebugPanel;

  constructor(url: string) {
    this.url = url;
    this.initializeHandlers();
    this.debugPanel = new NetworkDebugPanel();
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
      this.debugPanel.addMessage('in', event.data);
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
      if (this.isConnected && this.ws) {
        // Use native WebSocket ping
        const pingData = new Uint8Array([]);
        this.ws.send(pingData);
        this.debugPanel.addMessage('out', 'WebSocket Ping');
      }
    }, HEARTBEAT_INTERVAL);
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
      const messageStr = JSON.stringify(message);
      this.debugPanel.addMessage('out', messageStr);
      this.ws?.send(messageStr);
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
      this.debugPanel.addMessage('out', data);
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
