// public/js/services/websocketService.js

/**
 * WebsocketService handles the WebSocket connection and communication with the server.
 */
export class WebsocketService {
    constructor() {
        this.socket = null;
        this.listeners = {};
        this.connect();
    }

    /**
     * Establishes a WebSocket connection to the server.
     */
    connect() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${wsProtocol}//${window.location.host}/ws`; 
        console.log('Attempting to connect to WebSocket at:', url);
        this.socket = new WebSocket(url);

        // WebSocket open event
        this.socket.onopen = () => {
            console.log('WebSocket connection established');
            this.emit('open');
        };

        // WebSocket message event
        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.emit('message', data);
            } catch (err) {
                console.error('Error parsing WebSocket message:', err);
            }
        };

        // WebSocket error event
        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('error', error);
        };

        // WebSocket close event with reconnection logic
        this.socket.onclose = () => {
            console.log('WebSocket connection closed. Attempting to reconnect in 5 seconds...');
            this.emit('close');
            setTimeout(() => this.connect(), 5000); // Attempt reconnection after 5 seconds
        };
    }

    /**
     * Registers an event listener for a specific event type.
     * @param {string} event - The event type.
     * @param {function} callback - The callback function to execute when the event occurs.
     */
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    /**
     * Emits an event to all registered listeners.
     * @param {string} event - The event type.
     * @param {any} data - The data associated with the event.
     */
    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }

    /**
     * Sends data to the server via WebSocket.
     * @param {object} data - The data to send.
     */
    send(data) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(data));
        } else {
            console.warn('WebSocket is not open. Unable to send message:', data);
        }
    }
}
