import { jest } from '@jest/globals';

// Mock WebSocketService class
class WebSocketService {
  constructor(url) {
    this.url = url;
    this.socket = null;
  }

  connect() {
    this.socket = new WebSocket(this.url);
  }

  send(message) {
    this.socket.send(message);
  }

  onMessage(handler) {
    this.socket.addEventListener('message', handler);
  }

  close() {
    this.socket.close();
  }
}

describe('WebSocketService', () => {
  let websocketService;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = {
      send: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      close: jest.fn(),
    };
    global.WebSocket = jest.fn(() => mockWebSocket);

    websocketService = new WebSocketService('ws://localhost:8080');
  });

  test('connect should create a WebSocket connection', () => {
    websocketService.connect();
    expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:8080');
  });

  test('send should send a message through the WebSocket', () => {
    const message = 'Test message';
    websocketService.connect();
    websocketService.send(message);
    expect(mockWebSocket.send).toHaveBeenCalledWith(message);
  });

  test('onMessage should register a message handler', () => {
    const handler = jest.fn();
    websocketService.connect();
    websocketService.onMessage(handler);
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('message', handler);
  });

  test('close should close the WebSocket connection', () => {
    websocketService.connect();
    websocketService.close();
    expect(mockWebSocket.close).toHaveBeenCalled();
  });

  // Add more tests as needed based on WebSocketService functionality
});