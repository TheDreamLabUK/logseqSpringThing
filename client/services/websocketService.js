// Secure WebSocket service with improved error handling and security measures
export default class WebsocketService {
    // Event emitter implementation
    _events = new Map();

    // Rate limiting configuration
    messageQueue = [];
    messageRateLimit = 50; // messages per second
    messageTimeWindow = 1000; // 1 second
    lastMessageTime = 0;
    
    // Security configuration
    maxMessageSize = 1024 * 1024; // 1MB limit
    maxQueueSize = 100;
    validMessageTypes = new Set([
        'getInitialData',
        'graphUpdate',
        'audioData',
        'answer',
        'error',
        'ragflowResponse',
        'openaiResponse',
        'simulationModeSet',
        'fisheye_settings_updated',
        'completion',
        'position_update_complete',
        'graphData',
        'visualSettings',
        'materialSettings',
        'physicsSettings',
        'bloomSettings',
        'fisheyeSettings',
        'updateSettings',
        'settings_updated'
    ]);

    // WebSocket configuration
    socket = null;
    reconnectAttempts = 0;
    maxRetries = 3;
    retryDelay = 5000;
    isConnecting = false;

    constructor() {
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    // Event emitter methods
    on = (event, callback) => {
        if (!this._events.has(event)) {
            this._events.set(event, []);
        }
        this._events.get(event).push(callback);
    };

    off = (event, callback) => {
        if (!this._events.has(event)) return;
        const callbacks = this._events.get(event);
        const index = callbacks.indexOf(callback);
        if (index !== -1) {
            callbacks.splice(index, 1);
        }
    };

    emit = (event, ...args) => {
        if (!this._events.has(event)) return;
        const callbacks = this._events.get(event);
        callbacks.forEach(callback => {
            try {
                callback(...args);
            } catch (error) {
                console.error(`Error in event listener for ${event}:`, error);
            }
        });
    };

    // WebSocket URL generation
    getWebSocketUrl = () => {
        const isProduction = process.env.NODE_ENV === 'production';
        
        if (isProduction) {
            // In production, always use wss:// and fixed hostname
            // Port 4000 is handled by cloudflared tunnel
            return 'wss://www.visionflow.info/ws';
        }
        
        // In development, use current page's protocol and host
        // Port is handled by nginx in development
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}/ws`;
    };

    // Establish WebSocket connection
    connect = async () => {
        if (this.isConnecting || this.socket?.readyState === WebSocket.CONNECTING) {
            console.log('Connection attempt already in progress');
            return;
        }

        if (this.socket?.readyState === WebSocket.OPEN) {
            console.log('WebSocket already connected');
            return;
        }

        this.reconnectAttempts++;
        const url = this.getWebSocketUrl();
        console.log(`Attempting WebSocket connection (attempt ${this.reconnectAttempts}/${this.maxRetries})...`, {
            url,
            protocol: window.location.protocol,
            host: window.location.hostname,
            port: window.location.port || '(default)'
        });
        
        this.isConnecting = true;
        
        try {
            this.socket = new WebSocket(url);
            this.socket.binaryType = 'arraybuffer';

            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Connection timeout'));
                }, 10000); // 10 second timeout

                this.socket.onopen = () => {
                    clearTimeout(timeout);
                    console.log('WebSocket connection established');
                    this.reconnectAttempts = 0;
                    this.isConnecting = false;
                    this.emit('open');
                    this.processQueuedMessages();
                    resolve();
                };

                this.socket.onerror = (error) => {
                    console.error('WebSocket connection error:', {
                        message: error.message || 'Connection failed',
                        readyState: this.socket.readyState,
                        attempt: this.reconnectAttempts,
                        url
                    });
                    this.emit('error', error);
                };

                this.socket.onclose = (event) => {
                    clearTimeout(timeout);
                    console.log('WebSocket connection closed:', event);
                    this.emit('close');
                    this.isConnecting = false;

                    if (!event.wasClean && this.reconnectAttempts < this.maxRetries) {
                        console.log(`Connection failed. Retrying in ${this.retryDelay}ms...`, {
                            error: 'WebSocket error: Connection failed',
                            attempt: this.reconnectAttempts,
                            nextDelay: this.retryDelay
                        });
                        setTimeout(() => this.connect(), this.retryDelay);
                    } else if (this.reconnectAttempts >= this.maxRetries) {
                        console.error('Max reconnection attempts reached:', {
                            error: 'WebSocket error: Connection failed',
                            attempts: this.reconnectAttempts,
                            maxRetries: this.maxRetries
                        });
                        reject(new Error('Max reconnection attempts reached'));
                    }
                };
            });

            // Only set message handler after successful connection
            this.socket.onmessage = this.handleMessage;

        } catch (error) {
            this.isConnecting = false;
            console.error('Error initializing WebSocket:', error);
            throw error;
        }
    };

    handleMessage = async (event) => {
        try {
            if (event.data instanceof ArrayBuffer) {
                this.handleBinaryMessage(event.data);
            } else {
                await this.handleJsonMessage(event.data);
            }
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
            this.emit('error', { 
                type: 'processing_error', 
                message: error.message
            });
        }
    };

    handleBinaryMessage = (data) => {
        const view = new Float32Array(data);
        const header = view[0];
        const isInitialLayout = header >= 1.0;
        const timeStep = header % 1.0;
        
        // Process position data (skip header)
        const positions = [];
        for (let i = 1; i < view.length; i += 6) {
            if (i + 5 < view.length) {
                positions.push({
                    x: view[i],
                    y: view[i + 1],
                    z: view[i + 2],
                    vx: view[i + 3],
                    vy: view[i + 4],
                    vz: view[i + 5]
                });
            }
        }

        // Immediately emit position update event
        this.emit('gpuPositions', {
            isInitialLayout,
            timeStep,
            positions
        });

        // Also dispatch DOM event for compatibility
        window.dispatchEvent(new CustomEvent('binaryPositionUpdate', {
            detail: {
                isInitialLayout,
                timeStep,
                positions
            }
        }));
    };

    handleJsonMessage = async (data) => {
        let parsed;
        try {
            parsed = JSON.parse(data);
        } catch (error) {
            console.error('Failed to parse JSON message:', error);
            this.emit('error', { 
                type: 'parse_error', 
                message: 'Invalid JSON message received'
            });
            return;
        }

        if (data.length > this.maxMessageSize) {
            this.emit('error', { 
                type: 'size_error', 
                message: 'Message size exceeds limit'
            });
            return;
        }

        if (!parsed.type || !this.validMessageTypes.has(parsed.type)) {
            console.error('Invalid message type:', parsed.type);
            this.emit('error', { 
                type: 'validation_error', 
                message: 'Invalid message type'
            });
            return;
        }

        // First emit the raw message for any listeners that need it
        this.emit('message', parsed);
        
        // Then handle specific message types
        switch (parsed.type) {
            case 'graphData':
            case 'graphUpdate':
                if (parsed.graph_data) {
                    this.emit('graphUpdate', { graphData: parsed.graph_data });
                }
                break;

            case 'error':
                console.error('Server error:', parsed.message);
                this.emit('error', { 
                    type: 'server_error', 
                    message: parsed.message 
                });
                break;

            case 'settings_updated':
                window.dispatchEvent(new CustomEvent('settingsUpdated', {
                    detail: parsed.settings
                }));
                break;

            default:
                this.emit(parsed.type, parsed);
                break;
        }
    };

    send = (data) => {
        if (this.socket?.readyState === WebSocket.OPEN) {
            try {
                if (data instanceof ArrayBuffer) {
                    this.socket.send(data);
                } else {
                    this.socket.send(JSON.stringify(data));
                }
            } catch (error) {
                console.error('Error sending WebSocket message:', error);
                this.emit('error', { 
                    type: 'send_error', 
                    message: error.message 
                });
                this.queueMessage(data);
            }
        } else {
            this.queueMessage(data);
        }
    };

    queueMessage = (data) => {
        if (this.messageQueue.length >= this.maxQueueSize) {
            console.warn('Message queue full. Dropping message:', data);
            return;
        }
        this.messageQueue.push(data);
    };

    processQueuedMessages = () => {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;

        const now = Date.now();
        const messagesToSend = [];
        let messageCount = 0;

        while (this.messageQueue.length > 0 && 
               messageCount < this.messageRateLimit && 
               now - this.lastMessageTime >= this.messageTimeWindow / this.messageRateLimit) {
            messagesToSend.push(this.messageQueue.shift());
            this.lastMessageTime = now;
            messageCount++;
        }

        messagesToSend.forEach(message => this.send(message));
    };

    cleanup = () => {
        if (this.socket) {
            this.socket.onclose = null; // Prevent reconnection attempt
            this.socket.close();
            this.socket = null;
        }
        this.isConnecting = false;
        this.messageQueue = [];
        this._events.clear();
    };
}
