export class ChatManager {
  constructor(webSocket) {
    this.webSocket = webSocket;
    console.log('Initializing chat manager...');
  }

  sendMessage(message) {
    console.log('Sending message:', message);
    this.webSocket.send(JSON.stringify({ type: 'chat', message }));
  }

  handleWebSocketMessage(message) {
    const data = JSON.parse(message.data);
    if (data.type === 'chat') {
      this.updateChatDisplay(data.message);
    }
  }

  updateChatDisplay(message) {
    console.log('Received message:', message);
    // Implementation for updating the chat display
  }

  handleWebSocketError(error) {
    console.error('WebSocket error:', error);
  }
}