import { WebsocketService } from '../../public/js/services/websocketService';

describe('WebsocketService', () => {
  let websocketService;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = {
      send: jest.fn(),
      close: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn()
    };
    global.WebSocket = jest.fn(() => mockWebSocket);
    websocketService = new WebsocketService('wss://example.com');
  });

  test('WebsocketService initializes correctly', () => {
    expect(websocketService.url).toBe('wss://example.com');
    expect(websocketService.socket).toBe(mockWebSocket);
    expect(global.WebSocket).toHaveBeenCalledWith('wss://example.com');
  });

  test('connect establishes WebSocket connection', () => {
    websocketService.connect();
    expect(global.WebSocket).toHaveBeenCalledWith('wss://example.com');
  });

  test('disconnect closes WebSocket connection', () => {
    websocketService.disconnect();
    expect(mockWebSocket.close).toHaveBeenCalled();
  });

  test('send sends message through WebSocket', () => {
    const message = { type: 'test', data: 'Hello' };
    websocketService.send(message);
    expect(mockWebSocket.send).toHaveBeenCalledWith(JSON.stringify(message));
  });

  test('onMessage registers message handler', () => {
    const handler = jest.fn();
    websocketService.onMessage(handler);
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('message', expect.any(Function));
  });

  test('onOpen registers open handler', () => {
    const handler = jest.fn();
    websocketService.onOpen(handler);
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('open', handler);
  });

  test('onClose registers close handler', () => {
    const handler = jest.fn();
    websocketService.onClose(handler);
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('close', handler);
  });

  test('onError registers error handler', () => {
    const handler = jest.fn();
    websocketService.onError(handler);
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('error', handler);
  });

  test('removeMessageHandler removes message handler', () => {
    const handler = jest.fn();
    websocketService.onMessage(handler);
    websocketService.removeMessageHandler(handler);
    expect(mockWebSocket.removeEventListener).toHaveBeenCalledWith('message', expect.any(Function));
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
    expect(global.WebSocket).toHaveBeenCalledTimes(2); // Once in constructor, once in reconnect
  });

  test('handleMessage parses and calls handler with message data', () => {
    const handler = jest.fn();
    websocketService.onMessage(handler);
    const message = { data: JSON.stringify({ type: 'test', data: 'Hello' }) };
    mockWebSocket.addEventListener.mock.calls[0][1](message);
    expect(handler).toHaveBeenCalledWith({ type: 'test', data: 'Hello' });
  });
});
