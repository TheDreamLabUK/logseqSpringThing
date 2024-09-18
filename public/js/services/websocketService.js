export class WebsocketService {
  constructor(url) {
    this.url = url;
    this.socket = null;
    this.connect();
  }

  connect() {
    this.socket = new WebSocket(this.url);
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
    }
  }

  send(message) {
    if (this.socket && this.isConnected()) {
      this.socket.send(JSON.stringify(message));
    }
  }

  onMessage(handler) {
    if (this.socket) {
      this.socket.addEventListener('message', (event) => {
        this.handleMessage(event, handler);
      });
    }
  }

  onOpen(handler) {
    if (this.socket) {
      this.socket.addEventListener('open', handler);
    }
  }

  onClose(handler) {
    if (this.socket) {
      this.socket.addEventListener('close', handler);
    }
  }

  onError(handler) {
    if (this.socket) {
      this.socket.addEventListener('error', handler);
    }
  }

  removeMessageHandler(handler) {
    if (this.socket) {
      this.socket.removeEventListener('message', handler);
    }
  }

  isConnected() {
    return this.socket && this.socket.readyState === WebSocket.OPEN;
  }

  reconnect() {
    if (!this.isConnected()) {
      setTimeout(() => {
        this.connect();
      }, 1000);
    }
  }

  handleMessage(event, handler) {
    try {
      const parsedData = JSON.parse(event.data);
      handler(parsedData);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }
}
