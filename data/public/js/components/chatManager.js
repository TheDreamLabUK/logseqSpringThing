export class ChatManager {
    constructor(websocketService) {
        this.websocketService = websocketService;
        this.chatInput = null;
        this.sendButton = null;
        this.chatMessages = null;
    }

    initialize() {
        this.chatInput = document.getElementById('chat-input');
        this.sendButton = document.getElementById('send-button');
        this.chatMessages = document.getElementById('chat-messages');

        this.sendButton.addEventListener('click', this.sendMessage.bind(this));
        this.chatInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                this.sendMessage();
            }
        });
    }

    sendMessage() {
        const message = this.chatInput.value.trim();
        if (message) {
            this.websocketService.send({
                type: 'chatMessage',
                content: message
            });
            this.displayMessage('You', message);
            this.chatInput.value = '';
        }
    }

    displayMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.textContent = `${sender}: ${message}`;
        this.chatMessages.appendChild(messageElement);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    displayResponse(message) {
        this.displayMessage('AI', message);
    }
}
