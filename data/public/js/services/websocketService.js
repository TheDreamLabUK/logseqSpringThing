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
    maxAudioSize = 5 * 1024 * 1024; // 5MB limit
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
        'fisheye_settings_updated',  // Use underscore format consistently
        'completion',                // Add completion message type
        'position_update_complete'   // Add position update completion type
    ]);

    // WebSocket configuration
    socket = null;
    reconnectAttempts = 0;
    maxRetries = 3;
    retryDelay = 5000;
    
    // Audio configuration
    audioContext = null;
    audioQueue = [];
    isPlaying = false;
    audioInitialized = false;

    constructor() {
        // Initialize connection
        this.connect();
        
        // Clean up on page unload
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    // Event emitter methods
    on = (event, callback) => {
        if (!this._events.has(event)) {
            this._events.set(event, []);
        }
        this._events.get(event).push(callback);
    }

    off = (event, callback) => {
        if (!this._events.has(event)) return;
        const callbacks = this._events.get(event);
        const index = callbacks.indexOf(callback);
        if (index !== -1) {
            callbacks.splice(index, 1);
        }
    }

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
    }

    // Secure WebSocket URL generation
    getWebSocketUrl = () => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname;
        const port = window.location.port ? `:${window.location.port}` : '';
        const url = `${protocol}//${host}${port}/ws`;
        console.log('Generated WebSocket URL:', url);
        return url;
    }

    // Establish secure WebSocket connection
    connect = () => {
        const url = this.getWebSocketUrl();
        console.log('Attempting to connect to WebSocket at:', url);
        
        try {
            this.socket = new WebSocket(url);
            this.socket.binaryType = 'arraybuffer';  // Set binary type for position updates

            this.socket.onopen = () => {
                console.log('WebSocket connection established');
                this.reconnectAttempts = 0;
                this.emit('open');
                
                // Update connection status indicator
                const statusElement = document.getElementById('connection-status');
                if (statusElement) {
                    statusElement.textContent = 'Connected';
                    statusElement.className = 'connected';
                }
                
                // Request initial graph data and settings
                console.log('Requesting initial data');
                this.send({ type: 'getInitialData' });
                this.processQueuedMessages();
            };

            this.socket.onclose = (event) => {
                console.log('WebSocket connection closed:', event);
                
                // Update connection status indicator
                const statusElement = document.getElementById('connection-status');
                if (statusElement) {
                    statusElement.textContent = 'Disconnected';
                    statusElement.className = 'disconnected';
                }
                
                this.emit('close');
                this.reconnect();
            };

            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.emit('error', error);
            };

            this.socket.onmessage = this.handleMessage;
            
        } catch (error) {
            console.error('Error creating WebSocket connection:', error);
            this.emit('error', error);
        }
    }

    handleMessage = async (event) => {
        try {
            // Handle binary data (position updates)
            if (event.data instanceof ArrayBuffer) {
                // Log received buffer size
                console.log(`Received binary data of size: ${event.data.byteLength} bytes`);
                
                // Convert ArrayBuffer to Float32Array for position updates
                const positions = new Float32Array(event.data);
                const positionUpdates = [];
                
                // Extract header (first float32)
                const header = positions[0];
                const isInitialLayout = header >= 1.0;
                const timeStep = header % 1.0;
                console.log(`Received position update: isInitial=${isInitialLayout}, timeStep=${timeStep}`);
                
                // Process position data (skip header)
                for (let i = 1; i < positions.length; i += 6) {  // Start from 1 to skip header
                    if (i + 5 < positions.length) {  // Ensure we have complete node data
                        positionUpdates.push([
                            positions[i],     // x
                            positions[i + 1], // y
                            positions[i + 2], // z
                            positions[i + 3], // vx
                            positions[i + 4], // vy
                            positions[i + 5]  // vz
                        ]);
                    }
                }

                console.log(`Processed ${positionUpdates.length} node position updates`);

                // Emit the gpuPositions event with the updates
                this.emit('gpuPositions', { positions: positionUpdates });

                // Also dispatch the event for backward compatibility
                window.dispatchEvent(new CustomEvent('binaryPositionUpdate', {
                    detail: positionUpdates
                }));
                return;
            }

            // Handle JSON messages
            let data;
            try {
                data = JSON.parse(event.data);
            } catch (error) {
                console.error('Failed to parse JSON message:', error);
                console.error('Raw message:', event.data);
                this.emit('error', { 
                    type: 'parse_error', 
                    message: 'Invalid JSON message received',
                    details: error.message
                });
                return;
            }

            // Validate message size
            if (event.data.length > this.maxMessageSize) {
                console.error('Message exceeds size limit');
                this.emit('error', { 
                    type: 'size_error', 
                    message: 'Message size exceeds limit'
                });
                return;
            }

            // Validate message type
            if (!data.type || !this.validMessageTypes.has(data.type)) {
                console.error('Invalid message type:', data.type);
                this.emit('error', { 
                    type: 'validation_error', 
                    message: 'Invalid message type'
                });
                return;
            }

            // Process the validated message
            this.handleServerMessage(data);

        } catch (error) {
            console.error('Error processing WebSocket message:', error);
            console.error('Error stack:', error.stack);
            this.emit('error', { 
                type: 'processing_error', 
                message: error.message
            });
        }
    }

    reconnect = () => {
        if (this.reconnectAttempts < this.maxRetries) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxRetries}) in ${this.retryDelay / 1000} seconds...`);
            setTimeout(() => this.connect(), this.retryDelay);
        } else {
            console.error('Max reconnection attempts reached. Please refresh the page or check your connection.');
            this.emit('maxReconnectAttemptsReached');
        }
    }

    send = (data) => {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            if (data instanceof ArrayBuffer) {
                // Log binary data size before sending
                console.log(`Sending binary data of size: ${data.byteLength} bytes`);
                this.socket.send(data);
            } else {
                // Send JSON data
                console.log('Sending WebSocket message:', data);
                try {
                    this.socket.send(JSON.stringify(data));
                } catch (error) {
                    console.error('Error sending WebSocket message:', error);
                    this.emit('error', { type: 'send_error', message: error.message });
                }
            }
        } else {
            console.warn('WebSocket not open. Queuing message...');
            this.queueMessage(data);
        }
    }

    queueMessage = (data) => {
        if (this.messageQueue.length >= this.maxQueueSize) {
            console.warn('Message queue full. Dropping message:', data);
            return;
        }
        this.messageQueue.push(data);
    }

    processQueuedMessages = () => {
        const now = Date.now();
        const messagesToSend = [];
        let messageCount = 0;
        for (const message of this.messageQueue) {
            if (messageCount < this.messageRateLimit && now - this.lastMessageTime >= this.messageTimeWindow / this.messageRateLimit) {
                messagesToSend.push(message);
                this.lastMessageTime = now;
                messageCount++;
            } else {
                break; // Stop if rate limit is reached
            }
        }
        this.messageQueue.splice(0, messagesToSend.length);
        messagesToSend.forEach(message => this.send(message));
    }

    handleServerMessage = (data) => {
        console.log('Handling server message:', data);
        
        // Validate message type
        if (!this.validMessageTypes.has(data.type)) {
            console.warn('Received message with invalid type:', data.type);
            return;
        }
        
        // First emit the raw message for any listeners that need it
        this.emit('message', data);
        
        // Then handle specific message types
        switch (data.type) {
            case 'getInitialData':
                console.log('Received initial data:', data);
                if (data.graph_data) {
                    // Ensure graph data is properly structured
                    const graphData = {
                        nodes: Array.isArray(data.graph_data.nodes) ? data.graph_data.nodes : [],
                        edges: Array.isArray(data.graph_data.edges) ? data.graph_data.edges : [],
                        metadata: data.graph_data.metadata || {}
                    };
                    console.log('Emitting graph update with structured data:', graphData);
                    this.emit('graphUpdate', { graphData });
                }
                if (data.settings) {
                    console.log('Dispatching server settings:', data.settings);
                    if (data.settings.visualization) {
                        const viz = data.settings.visualization;
                        ['nodeColor', 'edgeColor', 'hologramColor'].forEach(key => {
                            if (viz[key]) {
                                let color = viz[key].replace(/['"]/g, '');
                                if (color.startsWith('0x')) {
                                    color = color.slice(2);
                                } else if (color.startsWith('#')) {
                                    color = color.slice(1);
                                }
                                color = color.padStart(6, '0');
                                viz[key] = '#' + color;
                            }
                        });
                    }
                    window.dispatchEvent(new CustomEvent('serverSettings', {
                        detail: data.settings
                    }));
                } else {
                    console.warn('No settings received in initial data');
                }
                break;
                
            case 'graphUpdate':
                console.log('Received graph update:', data.graph_data);
                if (data.graph_data) {
                    // Ensure graph data is properly structured
                    const graphData = {
                        nodes: Array.isArray(data.graph_data.nodes) ? data.graph_data.nodes : [],
                        edges: Array.isArray(data.graph_data.edges) ? data.graph_data.edges : [],
                        metadata: data.graph_data.metadata || {}
                    };
                    console.log('Emitting graph update with structured data:', graphData);
                    this.emit('graphUpdate', { graphData });
                }
                break;
                
            case 'audioData':
                this.handleAudioData(data.audio_data);
                break;
                
            case 'answer':
                this.emit('ragflowAnswer', data.answer);
                break;
                
            case 'error':
                console.error('Server error:', data.message);
                this.emit('error', { type: 'server_error', message: data.message });
                break;
                
            case 'ragflowResponse':
                this.handleRagflowResponse(data);
                break;
                
            case 'openaiResponse':
                this.emit('openaiResponse', data.response);
                break;
                
            case 'simulationModeSet':
                console.log('Simulation mode set:', data.mode);
                this.emit('simulationModeSet', data.mode);
                break;

            case 'fisheye_settings_updated':
                console.log('Fisheye settings updated:', data);
                const settings = {
                    enabled: data.fisheye_enabled,
                    strength: data.fisheye_strength,
                    focusPoint: [
                        data.fisheye_focus_x,
                        data.fisheye_focus_y,
                        data.fisheye_focus_z
                    ],
                    radius: data.fisheye_radius
                };
                window.dispatchEvent(new CustomEvent('fisheyeSettingsUpdated', {
                    detail: settings
                }));
                break;

            case 'completion':
                console.log('Received completion message:', data.message);
                this.emit('completion', data.message);
                break;

            case 'position_update_complete':
                console.log('Position update completed:', data.status);
                this.emit('positionUpdateComplete', data.status);
                break;
                
            default:
                console.warn('Unhandled message type:', data.type);
                break;
        }
    }

    handleRagflowResponse = (data) => {
        console.log('Handling RAGFlow response:', data);
        this.emit('ragflowAnswer', data.answer);
        if (data.audio) {
            const audioBlob = this.base64ToBlob(data.audio, 'audio/wav');
            this.handleAudioData(audioBlob);
        } else {
            console.warn('No audio data in RAGFlow response');
        }
    }

    base64ToBlob = (base64, mimeType) => {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }

    handleAudioData = async (audioBlob) => {
        if (!this.audioContext) {
            console.warn('AudioContext not initialized. Waiting for user interaction...');
            return;
        }

        try {
            console.log('Audio Blob size:', audioBlob.size);
            console.log('Audio Blob type:', audioBlob.type);
            const arrayBuffer = await audioBlob.arrayBuffer();
            console.log('ArrayBuffer size:', arrayBuffer.byteLength);
            const audioBuffer = await this.decodeWavData(arrayBuffer);
            this.audioQueue.push(audioBuffer);
            if (!this.isPlaying) {
                this.playNextAudio();
            }
        } catch (error) {
            console.error('Error processing audio data:', error);
            this.emit('error', { type: 'audio_processing_error', message: error.message });
        }
    }

    decodeWavData = async (wavData) => {
        return new Promise((resolve, reject) => {
            console.log('WAV data size:', wavData.byteLength);
            const dataView = new DataView(wavData.slice(0, 16));
            const header = new TextDecoder().decode(dataView);
            console.log('Header:', header);
            if (header !== 'RIFF') {
                console.error('Invalid WAV header:', header);
                return reject(new Error(`Invalid WAV header: ${header}`));
            }

            this.audioContext.decodeAudioData(
                wavData,
                (buffer) => {
                    console.log('Audio successfully decoded:', buffer);
                    resolve(buffer);
                },
                (error) => {
                    console.error('Error in decodeAudioData:', error);
                    reject(new Error(`Error decoding WAV data: ${error}`));
                }
            );
        });
    }

    playNextAudio = () => {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            return;
        }

        this.isPlaying = true;
        const audioBuffer = this.audioQueue.shift();
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        source.onended = () => this.playNextAudio();
        source.start();
    }

    initAudio = () => {
        if (this.audioInitialized) {
            console.log('Audio already initialized');
            return;
        }

        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.audioInitialized = true;
            console.log('AudioContext initialized');

            // Process any queued audio data
            if (this.audioQueue.length > 0 && !this.isPlaying) {
                this.playNextAudio();
            }
        } catch (error) {
            console.error('Failed to initialize AudioContext:', error);
            this.emit('error', { type: 'audio_init_error', message: error.message });
        }
    }

    setSimulationMode = (mode) => {
        this.send({
            type: 'setSimulationMode',
            mode: mode
        });
    }

    sendChatMessage = ({ message, useOpenAI }) => {
        this.send({
            type: 'chatMessage',
            message,
            use_openai: useOpenAI
        });
    }

    updateFisheyeSettings = (settings) => {
        console.log('Updating fisheye settings:', settings);
        const focus_point = settings.focusPoint || [0, 0, 0];
        this.send({
            type: 'updateFisheyeSettings',
            enabled: settings.enabled,
            strength: settings.strength,
            focus_point: focus_point,
            radius: settings.radius || 100.0
        });
    }

    cleanup = () => {
        if (this.socket) {
            this.socket.close();
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}
