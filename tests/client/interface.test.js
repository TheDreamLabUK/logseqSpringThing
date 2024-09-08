import { Interface } from '../../public/js/components/interface';

describe('Interface', () => {
  let interface;

  beforeEach(() => {
    // Mock the DOM elements that the Interface interacts with
    document.body.innerHTML = `
      <div id="chat-input"></div>
      <div id="send-button"></div>
      <div id="clear-chat-button"></div>
    `;
    interface = new Interface();
  });

  test('sendMessage should trigger when send button is clicked', () => {
    const mockSendMessage = jest.fn();
    interface.onSendMessage(mockSendMessage);

    const sendButton = document.getElementById('send-button');
    sendButton.click();

    expect(mockSendMessage).toHaveBeenCalled();
  });

  test('clearChat should trigger when clear chat button is clicked', () => {
    const mockClearChat = jest.fn();
    interface.onClearChat(mockClearChat);

    const clearChatButton = document.getElementById('clear-chat-button');
    clearChatButton.click();

    expect(mockClearChat).toHaveBeenCalled();
  });

  test('getInputValue should return the value of the chat input', () => {
    const chatInput = document.getElementById('chat-input');
    chatInput.value = 'Test message';

    expect(interface.getInputValue()).toBe('Test message');
  });

  test('clearInput should clear the chat input', () => {
    const chatInput = document.getElementById('chat-input');
    chatInput.value = 'Test message';

    interface.clearInput();

    expect(chatInput.value).toBe('');
  });

  // Add more tests as needed based on Interface functionality
});