// public/js/components/chatManager.js

/**
 * ChatManager handles the chat interface, sending user messages, and displaying AI responses.
 */
export class ChatManager {
    /**
     * Creates a new ChatManager instance.
     * @param {WebsocketService} websocketService - The WebSocket service instance.
     */
    constructor(websocketService) {
        this.websocketService = websocketService;
        this.chatInput = null;
        this.sendButton = null;
        this.chatMessages = null;
    }

    /**
     * Initializes the chat interface by setting up DOM elements and event listeners.
     */
    initialize() {
        this.chatInput = document.getElementById('chat-input');
        this.sendButton = document.getElementById('send-button');
        this.chatMessages = document.getElementById('chat-messages');

        // Event listener for send button
        this.sendButton.addEventListener('click', this.sendMessage.bind(this));

        // Event listener for 'Enter' key in the chat input
        this.chatInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Listen for RAGFlow AI responses
        window.addEventListener('ragflowAnswer', (event) => {
            const answer = event.detail;
            this.displayResponse(answer);
        });
    }

    /**
     * Sends a user message to the server via WebSocket.
     */
    sendMessage() {
        const message = this.chatInput.value.trim();
        if (message) {
            // Display user's message
            this.displayMessage('You', message);

            // Send the message to the server
            this.websocketService.send({
                type: 'ragflowQuery',
                question: message
            });

            // Clear the input field
            this.chatInput.value = '';
        }
    }

    /**
     * Displays a message in the chat interface.
     * @param {string} sender - The sender of the message ('You' or 'AI').
     * @param {string} message - The message content.
     */
    displayMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.style.marginBottom = '10px';
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        this.chatMessages.appendChild(messageElement);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    /**
     * Displays an AI response in the chat interface.
     * @param {string} message - The AI's response.
     */
    displayResponse(message) {
        this.displayMessage('AI', message);
    }
}
