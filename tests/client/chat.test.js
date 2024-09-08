// Mock ChatManager
class ChatManager {
  constructor() {
    this.chatHistory = [];
  }

  sendMessage(message) {
    this.chatHistory.push(message);
  }

  clearChat() {
    this.chatHistory = [];
  }

  getLastMessage() {
    return this.chatHistory[this.chatHistory.length - 1];
  }
}

describe('ChatManager', () => {
  let chatManager;

  beforeEach(() => {
    chatManager = new ChatManager();
  });

  test('sendMessage should add message to chat history', () => {
    const message = 'Hello, World!';
    chatManager.sendMessage(message);
    expect(chatManager.chatHistory).toContain(message);
  });

  test('clearChat should empty chat history', () => {
    chatManager.sendMessage('Test message');
    chatManager.clearChat();
    expect(chatManager.chatHistory).toHaveLength(0);
  });

  test('getLastMessage should return the most recent message', () => {
    const messages = ['First', 'Second', 'Third'];
    messages.forEach(msg => chatManager.sendMessage(msg));
    expect(chatManager.getLastMessage()).toBe('Third');
  });

  // Add more tests as needed based on ChatManager functionality
});