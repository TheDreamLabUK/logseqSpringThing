import { ChatManager } from '../../public/js/components/chatManager';

// Mock WebSocket
class MockWebSocket {
  constructor(url) {
    this.url = url;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
  }

  send(data) {
    // Mock send method
  }

  close() {
    // Mock close method
  }
}

describe('ChatManager', () => {
  let chatManager;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = new MockWebSocket('ws://localhost:8443');
    chatManager = new ChatManager(mockWebSocket);
  });

  test('should initialize properly', () => {
    expect(chatManager).toBeDefined();
  });

  test('should send a message via WebSocket', () => {
    const mockSend = jest.spyOn(mockWebSocket, 'send');
    chatManager.sendMessage('Hello, World!');

    expect(mockSend).toHaveBeenCalledWith(JSON.stringify({ type: 'chat', message: 'Hello, World!' }));
  });

  test('should handle incoming messages', () => {
    const mockMessage = { data: JSON.stringify({ type: 'chat', message: 'Hello from server!' }) };
    const mockUpdateChatDisplay = jest.fn();
    chatManager.updateChatDisplay = mockUpdateChatDisplay;

    chatManager.handleWebSocketMessage(mockMessage);

    expect(mockUpdateChatDisplay).toHaveBeenCalledWith('Hello from server!');
  });

  test('should handle WebSocket errors', () => {
    const mockConsoleError = jest.spyOn(console, 'error').mockImplementation();
    const mockError = new Error('WebSocket error');

    chatManager.handleWebSocketError(mockError);

    expect(mockConsoleError).toHaveBeenCalledWith('WebSocket error:', mockError);

    mockConsoleError.mockRestore();
  });
});
