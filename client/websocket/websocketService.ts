/**
 * WebSocket service for real-time communication
 */

import { 
  WebSocketMessage,
  MessageType,
  InitialDataMessage,
  BinaryPositionUpdateMessage,
  RequestInitialDataMessage,
  EnableBinaryUpdatesMessage,
  PingMessage,
} from '../core/types';
import { WS_RECONNECT_INTERVAL, WS_MESSAGE_QUEUE_SIZE } from '../core/constants';
import { createLogger } from '../core/utils';

const logger = createLogger('WebSocketService');

// Server configuration
const HEARTBEAT_INTERVAL = 15000; // 15 seconds
const BINARY_VERSION = 1.0;
const FLOATS_PER_NODE = 6; // x, y, z, vx, vy, vz
const VERSION_OFFSET = 1; // Skip version float

type MessageHandler = (data: any) => void;
type ErrorHandler = (error: Error) => void;
type ConnectionHandler = (connected: boolean) => void;

// Network debug panel
class NetworkDebugPanel {
  private container: HTMLDivElement;
  private messageList: HTMLUListElement;
  private maxMessages = 50;
  private binaryMessageCount = 0;

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
      display: none;
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

  addMessage(direction: 'in' | 'out', message: string | ArrayBuffer | Float32Array): void {
    const item = document.createElement('li');
    const timestamp = new Date().toISOString().split('T')[1].slice(0, -1);
    const arrow = direction === 'in' ? '←' : '→';
    let displayMessage: string;

    if (message instanceof ArrayBuffer || message instanceof Float32Array) {
      this.binaryMessageCount++;
      const byteLength = message instanceof ArrayBuffer ? message.byteLength : message.buffer.byteLength;
      const nodeCount = Math.floor((byteLength / 4 - VERSION_OFFSET) / FLOATS_PER_NODE);
      displayMessage = `Binary update #${this.binaryMessageCount} (${nodeCount} nodes, ${byteLength} bytes)`;
    } else if (typeof message === 'string') {
      try {
        const parsed = JSON.parse(message);
        displayMessage = `${parsed.type} ${JSON.stringify(parsed.data || {}, null, 0)}`;
      } catch {
        displayMessage = message;
      }
    } else {
      displayMessage = JSON.stringify(message, null, 0);
    }

    item.textContent = `${timestamp} ${arrow} ${displayMessage}`;
    item.style.marginBottom = '2px';
    item.style.wordBreak = 'break-all';

    this.messageList.insertBefore(item, this.messageList.firstChild);

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
  private connectionHandlers: ConnectionHandler[] = [];
  private isConnected: boolean = false;
  private debugPanel: NetworkDebugPanel | null = null;
  private expectedNodeCount: number = 0;

  constructor(url: string) {
    this.url = url;
    this.initializeHandlers();
    // Debug panel is now created but hidden by default
    this.debugPanel = new NetworkDebugPanel();
  }

  private initializeHandlers(): void {
    const messageTypes: MessageType[] = [
      'initialData',
      'requestInitialData',
      'binaryPositionUpdate',
      'enableBinaryUpdates',
      'ping',
      'pong'
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
      this.notifyConnectionHandlers(true);
      this.startHeartbeat();
      
      const requestInitialData: RequestInitialDataMessage = { type: 'requestInitialData' };
      this.send(requestInitialData);
      
      this.processMessageQueue();
    };

    this.ws.onclose = () => {
      logger.warn('WebSocket disconnected');
      this.isConnected = false;
      this.notifyConnectionHandlers(false);
      this.cleanup();
      this.scheduleReconnect();
    };

    this.ws.onerror = (event) => {
      logger.error('WebSocket error:', event);
      this.notifyErrorHandlers(new Error('WebSocket error occurred'));
    };

    this.ws.onmessage = (event) => {
      if (this.debugPanel) {
        this.debugPanel.addMessage('in', event.data);
      }
      this.handleMessage(event);
    };
  }

  private handleMessage(event: MessageEvent): void {
    try {
      if (event.data instanceof ArrayBuffer) {
        this.handleBinaryMessage(event.data);
      } else {
        this.handleJsonMessage(event.data);
      }
    } catch (error) {
      logger.error('Error handling message:', error);
      this.notifyErrorHandlers(new Error('Failed to process message'));
    }
  }

  private handleBinaryMessage(data: ArrayBuffer): void {
    const floatArray = new Float32Array(data);
    
    // Validate binary version
    const version = floatArray[0];
    if (version !== BINARY_VERSION) {
      logger.error(`Invalid binary version: ${version}`);
      return;
    }

    // Calculate and validate node count
    const nodeCount = Math.floor((floatArray.length - VERSION_OFFSET) / FLOATS_PER_NODE);
    if (nodeCount === 0 || nodeCount > this.expectedNodeCount) {
      logger.error(`Invalid node count: ${nodeCount}`);
      return;
    }

    // Pass the Float32Array directly to handlers
    this.notifyHandlers('binaryPositionUpdate', floatArray);
  }

  private handleJsonMessage(data: string): void {
    try {
      const message = JSON.parse(data) as WebSocketMessage;
      
      switch (message.type) {
        case 'initialData':
          this.handleInitialData(message);
          break;
        case 'ping':
          // Send pong response with same timestamp
          this.send({
            type: 'pong',
            timestamp: (message as PingMessage).timestamp
          });
          break;
        case 'pong':
          // Handle pong response if needed
          break;
        default:
          this.notifyHandlers(message.type, message);
      }
      
      if (this.debugPanel) {
        this.debugPanel.addMessage('in', JSON.stringify(message));
      }
    } catch (error) {
      logger.error('Error parsing WebSocket message:', error);
      this.notifyErrorHandlers(new Error('Failed to parse WebSocket message'));
    }
  }

  private handleInitialData(message: InitialDataMessage): void {
    this.notifyHandlers('initialData', message.data);
    
    const enableBinaryUpdates: EnableBinaryUpdatesMessage = { 
      type: 'enableBinaryUpdates',
      data: { enabled: true }
    };
    this.send(enableBinaryUpdates);
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

  private notifyConnectionHandlers(connected: boolean): void {
    this.connectionHandlers.forEach(handler => {
      try {
        handler(connected);
      } catch (error) {
        logger.error('Error in connection handler:', error);
      }
    });
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatInterval = window.setInterval(() => {
      if (this.isConnected && this.ws) {
        const ping: PingMessage = { 
          type: 'ping',
          timestamp: Date.now()
        };
        this.send(ping);
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
      if (this.debugPanel) {
        this.debugPanel.addMessage('out', messageStr);
      }
      this.ws?.send(messageStr);
    } catch (error) {
      logger.error('Error sending message:', error);
      this.notifyErrorHandlers(new Error('Failed to send message'));
    }
  }

  sendBinary(data: ArrayBuffer | Float32Array): void {
    if (!this.isConnected) {
      logger.warn('Cannot send binary data while disconnected');
      return;
    }

    try {
      if (this.debugPanel) {
        this.debugPanel.addMessage('out', data);
      }
      this.ws?.send(data instanceof Float32Array ? data.buffer : data);
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

  onConnectionChange(handler: ConnectionHandler): void {
    this.connectionHandlers.push(handler);
    handler(this.isConnected);
  }

  off(type: MessageType, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index !== -1) {
        handlers.splice(index, 1);
      }
    }
  }

  offError(handler: ErrorHandler): void {
    const index = this.errorHandlers.indexOf(handler);
    if (index !== -1) {
      this.errorHandlers.splice(index, 1);
    }
  }

  offConnectionChange(handler: ConnectionHandler): void {
    const index = this.connectionHandlers.indexOf(handler);
    if (index !== -1) {
      this.connectionHandlers.splice(index, 1);
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
