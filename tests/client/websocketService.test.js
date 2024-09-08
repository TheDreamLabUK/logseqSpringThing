import { WebSocketService } from '../../public/js/services/websocketService';

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
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('message', expect.any(Function));
  });

  test('close should close the WebSocket connection', () => {
    websocketService.connect();
    websocketService.close();
    expect(mockWebSocket.close).toHaveBeenCalled();
  });

  test('reconnect should attempt to reconnect after connection is closed', () => {
    jest.useFakeTimers();
    websocketService.connect();
    
    // Simulate a connection close
    const closeHandler = mockWebSocket.addEventListener.mock.calls.find(call => call[0] === 'close')[1];
    closeHandler();

    jest.runAllTimers();

    expect(global.WebSocket).toHaveBeenCalledTimes(2);
  });

  // Add more tests as needed based on WebSocketService functionality
});