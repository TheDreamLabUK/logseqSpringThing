import { ChatManager } from '../../public/js/components/chatManager';

describe('ChatManager', () => {
  let chatManager;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = {
      send: jest.fn()
    };
    chatManager = new ChatManager(mockWebSocket);
    console.log = jest.fn(); // Mock console.log
    console.error = jest.fn(); // Mock console.error
  });

  test('ChatManager initializes correctly', () => {
    expect(chatManager.webSocket).toBe(mockWebSocket);
    expect(console.log).toHaveBeenCalledWith('Initializing chat manager...');
  });

  test('sendMessage sends message through WebSocket', () => {
    const message = 'Test message';
    chatManager.sendMessage(message);

    expect(console.log).toHaveBeenCalledWith('Sending message:', message);
    expect(mockWebSocket.send).toHaveBeenCalledWith(JSON.stringify({ type: 'chat', message }));
  });

  test('handleWebSocketMessage processes chat messages correctly', () => {
    const updateChatDisplaySpy = jest.spyOn(chatManager, 'updateChatDisplay');
    const mockMessage = {
      data: JSON.stringify({ type: 'chat', message: 'Test chat message' })
    };

    chatManager.handleWebSocketMessage(mockMessage);

    expect(updateChatDisplaySpy).toHaveBeenCalledWith('Test chat message');
  });

  test('handleWebSocketMessage ignores non-chat messages', () => {
    const updateChatDisplaySpy = jest.spyOn(chatManager, 'updateChatDisplay');
    const mockMessage = {
      data: JSON.stringify({ type: 'other', message: 'Test other message' })
    };

    chatManager.handleWebSocketMessage(mockMessage);

    expect(updateChatDisplaySpy).not.toHaveBeenCalled();
  });

  test('updateChatDisplay logs received message', () => {
    const message = 'Test received message';
    chatManager.updateChatDisplay(message);

    expect(console.log).toHaveBeenCalledWith('Received message:', message);
  });

  test('handleWebSocketError logs error message', () => {
    const error = new Error('Test error');
    chatManager.handleWebSocketError(error);

    expect(console.error).toHaveBeenCalledWith('WebSocket error:', error);
  });
});