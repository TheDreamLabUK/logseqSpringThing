// public/js/components/chatManager.js

/**
 * ChatManager handles sending and receiving chat messages via WebSocket.
 */
export class ChatManager {
  /**
   * Creates a new ChatManager instance.
   * @param {WebsocketService} webSocket - The WebSocket service instance.
   */
  constructor(webSocket) {
    this.webSocket = webSocket;
    this.initChatListeners();
    this.setupUIListeners();
  }

  /**
   * Initializes WebSocket listeners for chat-related messages.
   */
  initChatListeners() {
    // Listen for chat messages from the server
    this.webSocket.on('message', (data) => {
      if (data.type === 'chat') {
        this.displayMessage('AI', data.message);
      }
    });
  }

  /**
   * Sets up UI event listeners for chat input and sending messages.
   */
  setupUIListeners() {
    const askButton = document.getElementById('ask-button');
    const chatInput = document.getElementById('chat-input');

    // Event listener for the "Ask" button
    askButton.addEventListener('click', () => this.sendMessage());

    // Event listener for pressing "Enter" key in the chat input
    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') this.sendMessage();
    });
  }

  /**
   * Sends a user message to the server via WebSocket.
   */
  sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const message = chatInput.value.trim();
    if (message === '') return;

    // Display user message in the chat window
    this.displayMessage('User', message);

    // Send the message to the server
    this.webSocket.send({
      type: 'chat',
      message: message
    });

    // Clear the chat input
    chatInput.value = '';
  }

  /**
   * Displays a message in the chat window.
   * @param {string} sender - The sender of the message ('User' or 'AI').
   * @param {string} message - The message content.
   */
  displayMessage(sender, message) {
    const chatWindow = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.className = 'chat-message';
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
}
