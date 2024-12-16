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
import { WS_RECONNECT_INTERVAL, WS_MESSAGE_QUEUE_SIZE, WS_URL, BINARY_VERSION, BINARY_HEADER_SIZE } from '../core/constants';
import { createLogger } from '../core/utils';
import { convertObjectKeysToCamelCase } from '../core/utils';
import { graphDataManager, settingsManager } from '../core/managers';

const logger = createLogger('WebSocketService');

// Server configuration
const HEARTBEAT_INTERVAL = 15000; // 15 seconds
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
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private messageQueue: Array<ArrayBuffer | string> = [];
  private isConnected = false;
  private debugPanel: NetworkDebugPanel | null = null;

  constructor() {
    this.connect();
    this.debugPanel = new NetworkDebugPanel();
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
    };

    this.ws.onclose = () => {
      this.isConnected = false;
      logger.info('WebSocket disconnected');
      this.scheduleReconnect();
    };

    this.ws.onerror = (error) => {
      logger.error('WebSocket error:', error);
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
        const positions = new Float32Array(buffer, BINARY_HEADER_SIZE);
        graphDataManager.updatePositions(positions);
      } else {
        logger.warn(`Unsupported binary message version: ${version}`);
      }
    } catch (error) {
      logger.error('Error handling binary message:', error);
    }
  }

  private handleJsonMessage(message: any): void {
    try {
      switch (message.type) {
        case 'settingsUpdated':
          // Convert snake_case to camelCase and update settings
          const settings = convertObjectKeysToCamelCase(message.data);
          settingsManager.updateSettingsFromServer(settings);
          break;
          
        case 'graphUpdated':
          graphDataManager.updateGraph(message.data);
          break;
          
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
          logger.warn(`Unknown message type: ${message.type}`);
      }
    } catch (error) {
      logger.error('Error handling JSON message:', error);
    }
  }

  private handleInitialData(message: InitialDataMessage): void {
    // Notify handlers
    const handlers = this.messageHandlers.get('initialData');
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(message.data);
        } catch (error) {
          logger.error(`Error in initialData handler:`, error);
        }
      });
    }
    
    const enableBinaryUpdates: EnableBinaryUpdatesMessage = { 
      type: 'enableBinaryUpdates',
      data: { enabled: true }
    };
    this.send(enableBinaryUpdates);
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

  public send(data: ArrayBuffer | string): void {
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
  }

  private messageHandlers: Map<MessageType, MessageHandler[]> = new Map();
  private errorHandlers: ErrorHandler[] = [];
  private connectionHandlers: ConnectionHandler[] = [];

  public on(type: MessageType, handler: MessageHandler): void {
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      handlers.push(handler);
    } else {
      this.messageHandlers.set(type, [handler]);
    }
  }

  public onError(handler: ErrorHandler): void {
    this.errorHandlers.push(handler);
  }

  public onConnectionChange(handler: ConnectionHandler): void {
    this.connectionHandlers.push(handler);
    handler(this.isConnected);
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
