// Secure WebSocket service with improved error handling and security measures
export default class WebsocketService {
    constructor() {
        // Rate limiting configuration
        this.messageQueue = [];
        this.messageRateLimit = 50; // messages per second
        this.messageTimeWindow = 1000; // 1 second
        this.lastMessageTime = 0;
        
        // Security configuration
        this.maxMessageSize = 1024 * 1024; // 1MB limit
        this.maxAudioSize = 5 * 1024 * 1024; // 5MB limit
        this.maxQueueSize = 100;
        this.validMessageTypes = new Set([
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
        this.socket = null;
        this.listeners = new Map();
        this.reconnectAttempts = 0;
        this.maxRetries = 3;
        this.retryDelay = 5000;
        
        // Audio configuration
        this.audioContext = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.audioInitialized = false;

        // Initialize connection
        this.connect();
        
        // Clean up on page unload
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    // Secure WebSocket URL generation
    getWebSocketUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname;
        const port = window.location.port ? `:${window.location.port}` : '';
        const url = `${protocol}//${host}${port}/ws`;
        console.log('Generated WebSocket URL:', url);
        return url;
    }

    // Establish secure WebSocket connection
    connect() {
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

            this.socket.onmessage = this.handleMessage.bind(this);
            
        } catch (error) {
            console.error('Error creating WebSocket connection:', error);
            this.emit('error', error);
        }
    }

    handleMessage = async (event) => {
        try {
            if (event.data instanceof ArrayBuffer) {
                // Handle binary position updates by passing directly to visualization
                window.dispatchEvent(new CustomEvent('binaryPositionUpdate', {
                    detail: event.data
                }));
                return;
            }

            // Handle JSON messages
            const data = JSON.parse(event.data);
            console.log('Received message:', data);
            this.handleServerMessage(data);
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
            console.error('Raw message:', event.data);
            this.emit('error', { 
                type: 'parse_error', 
                message: error.message, 
                rawData: event.data 
            });
        }
    }

    reconnect() {
        if (this.reconnectAttempts < this.maxRetries) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxRetries}) in ${this.retryDelay / 1000} seconds...`);
            setTimeout(() => this.connect(), this.retryDelay);
        } else {
            console.error('Max reconnection attempts reached. Please refresh the page or check your connection.');
            this.emit('maxReconnectAttemptsReached');
        }
    }

    send(data) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            if (data instanceof ArrayBuffer) {
                // Send binary data directly
                this.socket.send(data);
            } else {
                // Send JSON data
                console.log('Sending WebSocket message:', data);
                this.socket.send(JSON.stringify(data));
            }
        } else {
            console.warn('WebSocket is not open. Unable to send message:', data);
            this.emit('error', { type: 'send_error', message: 'WebSocket is not open' });
        }
    }

    on(event, callback) {
        if (typeof callback !== 'function') {
            console.error('Invalid callback provided for event:', event);
            return;
        }
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    off(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
        }
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => {
                if (typeof callback === 'function') {
                    try {
                        callback(data);
                    } catch (error) {
                        console.error(`Error in listener for event '${event}':`, error);
                    }
                } else {
                    console.warn(`Invalid listener for event '${event}':`, callback);
                }
            });
        }
    }

    handleServerMessage(data) {
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
                    console.log('Emitting graph update with:', data.graph_data);
                    this.emit('graphUpdate', { graphData: data.graph_data });
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
                    this.emit('graphUpdate', { graphData: data.graph_data });
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

    handleRagflowResponse(data) {
        console.log('Handling RAGFlow response:', data);
        this.emit('ragflowAnswer', data.answer);
        if (data.audio) {
            const audioBlob = this.base64ToBlob(data.audio, 'audio/wav');
            this.handleAudioData(audioBlob);
        } else {
            console.warn('No audio data in RAGFlow response');
        }
    }

    base64ToBlob(base64, mimeType) {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }

    async handleAudioData(audioBlob) {
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

    async decodeWavData(wavData) {
        return new Promise((resolve, reject) => {
            console.log('WAV data size:', wavData.byteLength);
            const dataView = new DataView(wavData);
            const firstBytes = Array.from(new Uint8Array(wavData.slice(0, 16)))
                .map(b => b.toString(16).padStart(2, '0')).join(' ');
            console.log('First 16 bytes:', firstBytes);

            const header = new TextDecoder().decode(wavData.slice(0, 4));
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

    playNextAudio() {
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

    initAudio() {
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

    setSimulationMode(mode) {
        this.send({
            type: 'setSimulationMode',
            mode: mode
        });
    }

    sendChatMessage({ message, useOpenAI }) {
        this.send({
            type: 'chatMessage',
            message,
            use_openai: useOpenAI
        });
    }

    updateFisheyeSettings(settings) {
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

    cleanup() {
        if (this.socket) {
            this.socket.close();
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}
