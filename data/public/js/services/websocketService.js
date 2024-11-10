// WebSocket service for handling real-time communication
export default class WebsocketService {
    constructor() {
        // Initialize with environment variables from .env_template
        this.maxRetries = parseInt(process.env.MAX_RETRIES) || 3;
        this.retryDelay = parseInt(process.env.RETRY_DELAY) || 5000;
        this.timeout = parseInt(process.env.API_CLIENT_TIMEOUT) || 30000;
        this.maxConcurrentRequests = parseInt(process.env.MAX_CONCURRENT_REQUESTS) || 5;
        
        // API Configuration
        this.perplexityApiKey = process.env.PERPLEXITY_API_KEY;
        this.perplexityModel = process.env.PERPLEXITY_MODEL;
        this.perplexityMaxTokens = parseInt(process.env.PERPLEXITY_MAX_TOKENS) || 4096;
        this.perplexityTemperature = parseFloat(process.env.PERPLEXITY_TEMPERATURE) || 0.5;
        this.perplexityTopP = parseFloat(process.env.PERPLEXITY_TOP_P) || 0.9;
        this.perplexityPresencePenalty = parseFloat(process.env.PERPLEXITY_PRESENCE_PENALTY) || 0.0;
        this.perplexityFrequencyPenalty = parseFloat(process.env.PERPLEXITY_FREQUENCY_PENALTY) || 1.0;
        this.perplexityApiUrl = process.env.PERPLEXITY_API_URL;
        
        this.openaiApiKey = process.env.OPENAI_API_KEY;
        this.openaiBaseUrl = process.env.OPENAI_BASE_URL;
        
        this.ragflowApiKey = process.env.RAGFLOW_API_KEY;
        this.ragflowBaseUrl = process.env.RAGFLOW_BASE_URL;

        // WebSocket setup
        this.socket = null;
        this.listeners = {};
        this.reconnectAttempts = 0;
        this.reconnectInterval = this.retryDelay;
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.audioQueue = [];
        this.isPlaying = false;
        
        // Force-directed parameters
        this.forceDirectedParams = {
            iterations: 100,
            attraction_strength: 0.01,
            damping: 0.9
        };
        
        this.connect();
    }

    getWebSocketUrl() {
        const host = window.location.hostname;
        return `wss://${host}:8443/ws`;  // Always use wss:// and port 8443 for SSL
    }

    connect() {
        const url = this.getWebSocketUrl();
        console.log('Attempting to connect to WebSocket at:', url);
        this.socket = new WebSocket(url);
        this.socket.binaryType = 'arraybuffer';  // Set binary type for position updates

        this.socket.onopen = () => {
            console.log('WebSocket connection established');
            this.reconnectAttempts = 0;
            this.emit('open');
            
            // Request initial graph data and settings
            console.log('Requesting initial data');
            this.send({ type: 'getInitialData' });
        };

        this.socket.onmessage = async (event) => {
            try {
                if (event.data instanceof ArrayBuffer) {
                    // Handle binary position updates
                    const positions = new Float32Array(event.data);
                    const positionArray = [];
                    
                    // Each position update contains 6 float values (x,y,z, vx,vy,vz)
                    for (let i = 0; i < positions.length; i += 6) {
                        positionArray.push({
                            position: {
                                x: positions[i],
                                y: positions[i + 1],
                                z: positions[i + 2]
                            },
                            velocity: {
                                x: positions[i + 3],
                                y: positions[i + 4],
                                z: positions[i + 5]
                            }
                        });
                    }
                    
                    window.dispatchEvent(new CustomEvent('nodePositionsUpdated', {
                        detail: positionArray
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
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('error', error);
        };

        this.socket.onclose = () => {
            console.log('WebSocket connection closed.');
            this.emit('close');
            this.reconnect();
        };
    }

    reconnect() {
        if (this.reconnectAttempts < this.maxRetries) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxRetries}) in ${this.reconnectInterval / 1000} seconds...`);
            setTimeout(() => this.connect(), this.reconnectInterval);
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

            case 'fisheyeSettingsUpdated':
                console.log('Fisheye settings updated:', data);
                const settings = {
                    enabled: data.enabled,
                    strength: data.strength,
                    focusPoint: data.focus_point,
                    radius: data.radius
                };
                window.dispatchEvent(new CustomEvent('fisheyeSettingsUpdated', {
                    detail: settings
                }));
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
            console.warn('AudioContext not initialized. Call initAudio() first.');
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
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log('AudioContext initialized');
        } else if (this.audioContext.state === 'suspended') {
            this.audioContext.resume().then(() => {
                console.log('AudioContext resumed');
            }).catch((error) => {
                console.error('Error resuming AudioContext:', error);
            });
        } else {
            console.log('AudioContext already initialized');
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
}
