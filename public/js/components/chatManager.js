export class ChatManager {
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