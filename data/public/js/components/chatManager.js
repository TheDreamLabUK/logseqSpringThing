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
        this.activeConversationId = null;
    }

    /**
     * Initializes the chat interface by setting up DOM elements and event listeners.
     */
    initialize() {
        this.chatInput = document.getElementById('chat-input');
        this.sendButton = document.getElementById('send-button');
        this.chatMessages = document.getElementById('chat-messages');

        // Event listener for send button
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Event listener for 'Enter' key in the chat input
        this.chatInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Initialize the chat conversation
        this.initializeConversation();

        // Set up WebSocket message handler
        this.websocketService.on('message', (data) => this.handleWebSocketMessage(data));
    }

    /**
     * Initializes a new conversation with the RAGflow service.
     */
    initializeConversation() {
        const userId = 'default_user'; // You might want to generate or obtain this dynamically
        console.log('Initializing conversation for user:', userId);
        this.websocketService.send({
            type: 'initChat',
            userId: userId
        });
    }

    /**
     * Handles the response from initializing a chat conversation.
     * @param {Object} data - The response data from the server.
     */
    handleInitChatResponse(data) {
        console.log('Received init chat response:', data);
        if (data.success) {
            this.activeConversationId = data.conversationId;
            console.log("Chat initialized with conversation ID:", this.activeConversationId);
            this.displayMessage('System', "Chat initialized successfully. You can start chatting now.");
        } else {
            console.error("Failed to initialize chat");
            this.displayMessage('System', "Failed to initialize chat. Please try again.");
        }
    }

    /**
     * Sends a user message to the server via WebSocket.
     * @param {boolean} quote - Optional. Whether to quote the message.
     * @param {string[]} docIds - Optional. Array of document IDs to restrict the search.
     * @param {boolean} stream - Optional. Whether to stream the response.
     */
    sendMessage(quote = false, docIds = null, stream = false) {
        const message = this.chatInput.value.trim();
        if (message) {
            if (!this.activeConversationId) {
                console.error("No active conversation. Please wait for chat initialization.");
                this.displayMessage('System', "Chat is not initialized yet. Please wait.");
                return;
            }

            console.log('Sending message:', message);
            // Display user's message
            this.displayMessage('You', message);

            // Send the message to the server
            this.websocketService.send({
                type: 'ragflowQuery',
                conversationId: this.activeConversationId,
                message: message,
                quote: quote,
                docIds: docIds,
                stream: stream
            });

            // Clear the input field
            this.chatInput.value = '';
        }
    }

    /**
     * Displays a message in the chat interface.
     * @param {string} sender - The sender of the message ('You', 'AI', or 'System').
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

    /**
     * Handles the RAGflow response from the server.
     * @param {Object} data - The response data from the server.
     */
    handleRagflowResponse(data) {
        console.log('Received RAGflow response:', data);
        if (data.retcode === 0) {
            const response = data.data.message[0].content;
            console.log("Received RAGFlow answer:", response);
            this.displayResponse(response);
        } else {
            console.error("RAGFlow error:", data.data.message[0].content);
            this.displayMessage('System', "Error in RAGFlow response: " + data.data.message[0].content);
        }
    }

    /**
     * Retrieves the chat history for the current conversation.
     */
    getChatHistory() {
        if (this.activeConversationId) {
            console.log('Requesting chat history for conversation:', this.activeConversationId);
            this.websocketService.send({
                type: 'chatHistory',
                conversationId: this.activeConversationId
            });
        } else {
            console.error("No active conversation. Cannot retrieve chat history.");
        }
    }

    /**
     * Handles the chat history response from the server.
     * @param {Object} data - The chat history data from the server.
     */
    handleChatHistoryResponse(data) {
        console.log('Received chat history response:', data);
        if (data.retcode === 0) {
            const messages = data.data.message;
            // Clear existing messages
            this.chatMessages.innerHTML = '';
            // Display each message in the history
            messages.forEach(msg => {
                this.displayMessage(msg.role === 'user' ? 'You' : 'AI', msg.content);
            });
        } else {
            console.error("Failed to retrieve chat history:", data.data.message[0].content);
            this.displayMessage('System', "Failed to retrieve chat history. Please try again.");
        }
    }

    /**
     * Handles incoming WebSocket messages.
     * @param {Object} data - The message data from the WebSocket.
     */
    handleWebSocketMessage(data) {
        console.log("Received WebSocket message:", data);
        switch (data.type) {
            case 'chatInitResponse':
                this.handleInitChatResponse(data);
                break;
            case 'ragflowResponse':
                this.handleRagflowResponse(data.data);
                break;
            case 'chatHistoryResponse':
                this.handleChatHistoryResponse(data.data);
                break;
            default:
                console.warn("Unhandled WebSocket message type:", data.type);
        }
    }
}
