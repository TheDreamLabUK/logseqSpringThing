// public/js/components/chatManager.js

export class ChatManager {
  constructor(websocketService) {
    this.websocketService = websocketService;
    this.chatInput = null;
    this.sendButton = null;
    this.chatMessages = null;
    this.aiToggle = null;
    this.isChatReady = false;
    this.useOpenAI = false;
  }

  initialize() {
    this.chatInput = document.getElementById('chat-input');
    this.sendButton = document.getElementById('send-button');
    this.chatMessages = document.getElementById('chat-messages');
    this.aiToggle = document.getElementById('ai-toggle');

    this.sendButton.addEventListener('click', () => this.sendMessage());
    this.chatInput.addEventListener('keypress', (event) => {
      if (event.key === 'Enter') {
        this.sendMessage();
      }
    });

    if (this.aiToggle) {
      this.aiToggle.addEventListener('change', (event) => {
        this.useOpenAI = event.target.checked;
        console.log(`Switched to ${this.useOpenAI ? 'OpenAI' : 'RAGFlow'}`);
      });
    } else {
      console.warn('AI toggle not found');
    }

    this.websocketService.on('open', () => this.handleChatReady());
    this.websocketService.on('message', (data) => this.handleServerMessage(data));
    this.websocketService.on('error', (error) => this.handleError(error));

    // Initialize audio
    this.websocketService.initAudio();
  }

  handleChatReady() {
    console.log("Chat is ready");
    this.isChatReady = true;
    this.displayMessage('System', "Chat is ready. You can start chatting now.");
  }

  sendMessage() {
    const message = this.chatInput.value.trim();
    if (message) {
      if (!this.isChatReady) {
        console.error("Chat is not ready yet. Please wait.");
        this.displayMessage('System', "Chat is not ready yet. Please wait.");
        return;
      }

      console.log('Sending message:', message);
      this.displayMessage('You', message);
      
      if (this.useOpenAI) {
        this.websocketService.sendOpenAIQuery(message);
      } else {
        this.websocketService.sendRagflowQuery(message);
      }
      
      this.chatInput.value = '';
    }
  }

  displayMessage(sender, message) {
    const messageElement = document.createElement('div');
    messageElement.style.marginBottom = '10px';
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    this.chatMessages.appendChild(messageElement);
    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
  }

  handleServerMessage(data) {
    console.log("Received server message:", data);
    if (data.data) {
      const answer = data.data.answer;
      const reference = data.data.reference || [];
      const audioBase64 = data.data.audio;

      if (audioBase64) {
        // Decode Base64 to binary
        const binaryAudio = atob(audioBase64);
        const len = binaryAudio.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
          bytes[i] = binaryAudio.charCodeAt(i);
        }

        // Create Blob and Audio URL
        const blob = new Blob([bytes.buffer], { type: 'audio/wav' });
        const audioURL = URL.createObjectURL(blob);
        
        // Play the audio
        const audio = new Audio(audioURL);
        audio.play();
      } else {
        console.warn('Audio data not found in JSON response.');
      }

      this.displayMessage('AI', answer);
    } else {
      console.error('Invalid RAGFlow response:', data);
      this.handleError({ message: 'Invalid response from server' });
    }
  }

  handleError(error) {
    console.error("Error:", error);
    this.displayMessage('System', `Error: ${error.message || 'Unknown error occurred'}`);
  }
}
