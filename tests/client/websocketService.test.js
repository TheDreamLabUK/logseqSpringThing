import { initWebSocket, sendWebSocketMessage, closeWebSocketConnection } from '../../public/js/services/websocketService';

describe('WebSocket Service', () => {
  test('initWebSocket function exists', () => {
    expect(typeof initWebSocket).toBe('function');
  });

  test('initWebSocket function returns an object with send and close methods', () => {
    const socket = initWebSocket('ws://example.com');
    expect(typeof socket.send).toBe('function');
    expect(typeof socket.close).toBe('function');
  });

  test('sendWebSocketMessage function exists', () => {
    expect(typeof sendWebSocketMessage).toBe('function');
  });

  test('sendWebSocketMessage function returns true', () => {
    const mockSocket = { send: jest.fn() };
    expect(sendWebSocketMessage(mockSocket, 'test message')).toBe(true);
    expect(mockSocket.send).toHaveBeenCalledWith('test message');
  });

  test('closeWebSocketConnection function exists', () => {
    expect(typeof closeWebSocketConnection).toBe('function');
  });

  test('closeWebSocketConnection function returns true', () => {
    const mockSocket = { close: jest.fn() };
    expect(closeWebSocketConnection(mockSocket)).toBe(true);
    expect(mockSocket.close).toHaveBeenCalled();
  });
});
