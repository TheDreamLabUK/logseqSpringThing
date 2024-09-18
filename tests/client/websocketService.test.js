import { WebsocketService } from '../../public/js/services/websocketService';

describe('WebsocketService', () => {
  let websocketService;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = {
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      send: jest.fn(),
      close: jest.fn(),
      readyState: WebSocket.OPEN,
    };

    global.WebSocket = jest.fn(() => mockWebSocket);
    websocketService = new WebsocketService('ws://localhost:8080');
    websocketService.socket = mockWebSocket; // Ensure the socket is set
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('constructor initializes WebSocket connection', () => {
    expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:8080');
  });

  test('connect method creates a new WebSocket connection', () => {
    websocketService.connect();
    expect(global.WebSocket).toHaveBeenCalledTimes(2);
  });

  test('disconnect method closes the WebSocket connection', () => {
    websocketService.disconnect();
    expect(mockWebSocket.close).toHaveBeenCalled();
  });

  test('send method sends a message through WebSocket', () => {
    const message = { type: 'test', data: 'Hello' };
    websocketService.send(message);
    expect(mockWebSocket.send).toHaveBeenCalledWith(JSON.stringify(message));
  });

  test('onMessage adds a message event listener', () => {
    const handler = jest.fn();
    websocketService.onMessage(handler);
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('message', expect.any(Function));
  });

  test('removeMessageHandler removes a message event listener', () => {
    const handler = jest.fn();
    websocketService.removeMessageHandler(handler);
    expect(mockWebSocket.removeEventListener).toHaveBeenCalledWith('message', handler);
  });

  test('isConnected returns true when WebSocket is open', () => {
    mockWebSocket.readyState = WebSocket.OPEN;
    expect(websocketService.isConnected()).toBe(true);
  });

  test('isConnected returns false when WebSocket is not open', () => {
    mockWebSocket.readyState = WebSocket.CLOSED;
    expect(websocketService.isConnected()).toBe(false);
  });

  test('reconnect attempts to reconnect when disconnected', () => {
    jest.useFakeTimers();
    mockWebSocket.readyState = WebSocket.CLOSED;
    websocketService.reconnect();
    jest.runAllTimers();
    expect(global.WebSocket).toHaveBeenCalledTimes(2); // Changed from 2 to 1
  });

  test('handleMessage parses and calls handler with message data', () => {
    const handler = jest.fn();
    const mockEvent = { data: JSON.stringify({ type: 'test', data: 'Hello' }) };
    websocketService.handleMessage(mockEvent, handler);
    expect(handler).toHaveBeenCalledWith({ type: 'test', data: 'Hello' });
  });
});
