// public/js/services/websocketService.js

/**
 * WebsocketService handles WebSocket connections, messaging, and event management.
 */
export class WebsocketService {
  /**
   * Creates a new WebsocketService instance.
   * @param {string} url - The WebSocket server URL.
   */
  constructor(url) {
    this.url = url;
    this.socket = null;
    this.listeners = {};
    this.connect();
  }

  /**
   * Establishes a WebSocket connection to the server.
   */
  connect() {
    this.socket = new WebSocket(this.url);

    // Event handler for when the connection is opened
    this.socket.onopen = () => {
      console.log('WebSocket connection established');
      this.emit('open');
    };

    // Event handler for incoming messages
    this.socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.emit('message', data);
    };

    // Event handler for errors
    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    };

    // Event handler for when the connection is closed
    this.socket.onclose = () => {
      console.log('WebSocket connection closed. Attempting to reconnect...');
      this.emit('close');
      // Attempt to reconnect after 5 seconds
      setTimeout(() => this.connect(), 5000);
    };
  }

  /**
   * Registers an event listener for a specific event.
   * @param {string} event - The event name ('open', 'message', 'error', 'close').
   * @param {function} callback - The callback function to execute when the event occurs.
   */
  on(event, callback) {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(callback);
  }

  /**
   * Emits an event to all registered listeners.
   * @param {string} event - The event name.
   * @param {any} data - The data to pass to the listeners.
   */
  emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => callback(data));
    }
  }

  /**
   * Sends a message to the WebSocket server.
   * @param {object} data - The data to send.
   */
  send(data) {
    if (this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket is not open. Unable to send message:', data);
    }
  }

  /**
   * Disconnects the WebSocket connection gracefully.
   */
  disconnect() {
    if (this.socket) {
      this.socket.close();
    }
  }
}
