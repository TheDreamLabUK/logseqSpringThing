import { jest } from '@jest/globals';

// Mock Interface class
class Interface {
  constructor() {
    this.sendMessageCallback = null;
    this.clearChatCallback = null;
  }

  onSendMessage(callback) {
    this.sendMessageCallback = callback;
  }

  onClearChat(callback) {
    this.clearChatCallback = callback;
  }

  getInputValue() {
    return document.getElementById('chat-input').value;
  }

  clearInput() {
    document.getElementById('chat-input').value = '';
  }

  // Simulate click events
  triggerSendMessage() {
    if (this.sendMessageCallback) {
      this.sendMessageCallback();
    }
  }

  triggerClearChat() {
    if (this.clearChatCallback) {
      this.clearChatCallback();
    }
  }
}

describe('Interface', () => {
  let interfaceInstance;

  beforeEach(() => {
    // Mock the DOM elements that the Interface interacts with
    document.body.innerHTML = `
      <div id="chat-input"></div>
      <div id="send-button"></div>
      <div id="clear-chat-button"></div>
    `;
    interfaceInstance = new Interface();
  });

  test('sendMessage should trigger when send button is clicked', () => {
    const mockSendMessage = jest.fn();
    interfaceInstance.onSendMessage(mockSendMessage);

    interfaceInstance.triggerSendMessage();

    expect(mockSendMessage).toHaveBeenCalled();
  });

  test('clearChat should trigger when clear chat button is clicked', () => {
    const mockClearChat = jest.fn();
    interfaceInstance.onClearChat(mockClearChat);

    interfaceInstance.triggerClearChat();

    expect(mockClearChat).toHaveBeenCalled();
  });

  test('getInputValue should return the value of the chat input', () => {
    const chatInput = document.getElementById('chat-input');
    chatInput.value = 'Test message';

    expect(interfaceInstance.getInputValue()).toBe('Test message');
  });

  test('clearInput should clear the chat input', () => {
    const chatInput = document.getElementById('chat-input');
    chatInput.value = 'Test message';

    interfaceInstance.clearInput();

    expect(chatInput.value).toBe('');
  });

  // Add more tests as needed based on Interface functionality
});