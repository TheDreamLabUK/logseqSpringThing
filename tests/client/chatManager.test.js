import { ChatManager } from '../../public/js/components/chatManager';

describe('Chat Manager', () => {
  let chatManager;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = { send: jest.fn() };
    chatManager = new ChatManager(mockWebSocket);
  });

  test('ChatManager can be instantiated', () => {
    expect(chatManager).toBeInstanceOf(ChatManager);
  });

  test('sendMessage method exists', () => {
    expect(typeof chatManager.sendMessage).toBe('function');
  });

  test('sendMessage method sends a message via WebSocket', () => {
    const message = 'test message';
    chatManager.sendMessage(message);
    expect(mockWebSocket.send).toHaveBeenCalledWith(JSON.stringify({ type: 'chat', message }));
  });

  test('handleWebSocketMessage method exists', () => {
    expect(typeof chatManager.handleWebSocketMessage).toBe('function');
  });

  test('handleWebSocketMessage method processes incoming messages', () => {
    const mockUpdateChatDisplay = jest.fn();
    chatManager.updateChatDisplay = mockUpdateChatDisplay;

    const message = { data: JSON.stringify({ type: 'chat', message: 'Hello!' }) };
    chatManager.handleWebSocketMessage(message);

    expect(mockUpdateChatDisplay).toHaveBeenCalledWith('Hello!');
  });

  test('handleWebSocketError method exists', () => {
    expect(typeof chatManager.handleWebSocketError).toBe('function');
  });

  test('handleWebSocketError method logs errors', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    const error = new Error('Test error');
    chatManager.handleWebSocketError(error);
    expect(consoleSpy).toHaveBeenCalledWith('WebSocket error:', error);
    consoleSpy.mockRestore();
  });
});