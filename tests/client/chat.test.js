// chat.test.js

const ChatManager = require('../../public/js/components/chatManager');
const WebSocket = require('ws');

describe('ChatManager', () => {
  let chatManager;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = new WebSocket('ws://localhost:8443');
    chatManager = new ChatManager(mockWebSocket);
  });

  afterEach(() => {
    mockWebSocket.close();
  });

  test('should initialize properly', () => {
    expect(chatManager).toBeDefined();
    expect(chatManager.websocket).toBeDefined();
  });

  test('should send a message via WebSocket', () => {
    const sendSpy = jest.spyOn(mockWebSocket, 'send');
    const message = 'Hello, RAGFlow!';
    chatManager.sendMessage(message);
    expect(sendSpy).toHaveBeenCalledWith(
      JSON.stringify({
        type: 'chatMessage',
        content: message,
      })
    );
  });

  test('should handle incoming messages', () => {
    const mockMessage = {
      data: JSON.stringify({
        type: 'chatResponse',
        content: 'Response from RAGFlow',
      }),
    };

    const onMessageReceivedSpy = jest.fn();
    chatManager.onMessageReceived = onMessageReceivedSpy;

    mockWebSocket.onmessage(mockMessage);

    expect(onMessageReceivedSpy).toHaveBeenCalledWith('Response from RAGFlow');
  });

  test('should handle WebSocket errors', () => {
    const errorEvent = new Error('WebSocket error');
    const onErrorSpy = jest.fn();
    chatManager.onError = onErrorSpy;

    mockWebSocket.onerror(errorEvent);

    expect(onErrorSpy).toHaveBeenCalledWith(errorEvent);
  });
});
