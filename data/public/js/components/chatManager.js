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

        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
    }

    sendMessage() {
        const message = this.chatInput.value.trim();
        if (message) {
            this.websocketService.send({
                type: 'chatMessage',
                message: message
            });
            this.addMessageToChat('You', message);
            this.chatInput.value = '';
        }
    }

    displayResponse(message) {
        this.addMessageToChat('AI', message);
    }

    addMessageToChat(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        this.chatMessages.appendChild(messageElement);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
}
