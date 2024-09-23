import { ChatManager } from '../../public/js/components/chatManager';

describe('ChatManager', () => {
  let chatManager;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = {
      send: jest.fn()
    };
    console.log = jest.fn(); // Mock console.log
    chatManager = new ChatManager(mockWebSocket);
  });

  test('ChatManager initializes correctly', () => {
    const consoleSpy = jest.spyOn(console, 'log');
    const chatManager = new ChatManager(mockWebSocket);
    expect(consoleSpy).toHaveBeenCalledWith('Initializing chat manager...');
  });
  
  test('sendMessage sends message through WebSocket', () => {
    const message = 'Test message';
    chatManager.sendMessage(message);

    expect(console.log).toHaveBeenCalledWith('Sending message:', message);
    expect(mockWebSocket.send).toHaveBeenCalledWith(JSON.stringify({ type: 'chat', message }));
  });

  test('handleWebSocketMessage processes chat messages correctly', () => {
    const mockMessage = {
      data: JSON.stringify({ type: 'chat', message: 'Test chat message' })
    };

    chatManager.handleWebSocketMessage(mockMessage);

    expect(console.log).toHaveBeenCalledWith('Received message:', 'Test chat message');
  });

  test('handleWebSocketMessage ignores non-chat messages', () => {
    const mockMessage = {
      data: JSON.stringify({ type: 'other', message: 'Test other message' })
    };

    chatManager.handleWebSocketMessage(mockMessage);

    expect(console.log).not.toHaveBeenCalledWith('Received message:', 'Test other message');
  });

  test('handleWebSocketError logs error message', () => {
    console.error = jest.fn(); // Mock console.error
    const error = new Error('Test error');
    chatManager.handleWebSocketError(error);

    expect(console.error).toHaveBeenCalledWith('WebSocket error:', error);
  });
});