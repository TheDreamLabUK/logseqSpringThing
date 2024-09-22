export class WebsocketService {
    constructor() {
        this.socket = null;
        this.listeners = {};
        this.connect();
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        const url = `${protocol}//${host}/ws`;

        console.log('Attempting to connect to WebSocket at:', url);
        this.socket = new WebSocket(url);

        this.socket.onopen = () => {
            console.log('WebSocket connection established');
            this.emit('open');
        };

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.emit('message', data);
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('error', error);
        };

        this.socket.onclose = () => {
            console.log('WebSocket connection closed. Attempting to reconnect...');
            this.emit('close');
            setTimeout(() => this.connect(), 5000);
        };
    }

    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }

    send(data) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(data));
        } else {
            console.warn('WebSocket is not open. Unable to send message:', data);
        }
    }
}
