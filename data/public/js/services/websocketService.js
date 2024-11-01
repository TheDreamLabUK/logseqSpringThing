// public/js/services/websocketService.js

import pako from 'pako';  // Static import instead of dynamic

/**
 * WebsocketService handles the WebSocket connection and communication with the server.
 */
class WebsocketService {
    constructor() {
        this.socket = null;
        this.listeners = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 5000; // 5 seconds
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.audioQueue = [];
        this.isPlaying = false;
        this.connect();
    }

    /**
     * Establishes a WebSocket connection to the server.
     */
    connect() {
        const url = 'wss://192.168.0.51:8443/ws';
        console.log('Attempting to connect to WebSocket at:', url);
        this.socket = new WebSocket(url);
        this.socket.binaryType = 'arraybuffer';  // Set to handle binary data

        this.socket.onopen = () => {
            console.log('WebSocket connection established');
            this.reconnectAttempts = 0;
            this.emit('open');
            
            // Request initial graph data when connection is established
            this.send({ type: 'getInitialData' });
        };

        this.socket.onmessage = async (event) => {
            try {
                let data;
                if (event.data instanceof ArrayBuffer) {
                    // Handle compressed binary data
                    const decompressed = this.decompressMessage(event.data);
                    data = JSON.parse(decompressed);
                    console.log('Received and decompressed binary message:', data);
                } else {
                    // Handle regular JSON messages
                    data = JSON.parse(event.data);
                    console.log('Received JSON message:', data);
                }
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

    /**
     * Decompresses a binary message using Pako.
     * @param {ArrayBuffer} compressed - The compressed binary data.
     * @returns {string} The decompressed string.
     */
    decompressMessage(compressed) {
        try {
            const decompressed = pako.inflate(new Uint8Array(compressed), { to: 'string' });
            return decompressed;
        } catch (error) {
            console.error('Error decompressing message:', error);
            throw error;
        }
    }

    // Rest of the file remains exactly the same...
    setSimulationMode(mode) {
        this.send({
            type: 'set_simulation_mode',
            mode: mode
        });
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

    reconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${this.reconnectInterval / 1000} seconds...`);
            setTimeout(() => this.connect(), this.reconnectInterval);
        } else {
            console.error('Max reconnection attempts reached. Please refresh the page or check your connection.');
            this.emit('maxReconnectAttemptsReached');
        }
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
            console.log('Sending WebSocket message:', data);
            this.socket.send(JSON.stringify(data));
        } else {
            console.warn('WebSocket is not open. Unable to send message:', data);
            this.emit('error', { type: 'send_error', message: 'WebSocket is not open' });
        }
    }

    sendRagflowQuery(message, quote = false, docIds = null) {
        this.send({
            type: 'ragflowQuery',
            message,
            quote,
            docIds
        });
    }

    sendOpenAIQuery(message) {
        this.send({
            type: 'openaiQuery',
            message
        });
    }

    sendChatMessage({ message, useOpenAI }) {
        if (useOpenAI) {
            this.sendOpenAIQuery(message);
        } else {
            this.sendRagflowQuery(message);
        }
    }

    handleServerMessage(data) {
        console.log('Handling server message:', data);
        
        // First emit the raw message for any listeners that need it
        this.emit('message', data);
        
        // Then handle specific message types
        switch (data.type) {
            case 'initial_data':
                console.log('Received initial graph data');
                this.emit('graphUpdate', data.graph_data);
                break;
                
            case 'graph_update':
                console.log('Received graph update');
                this.emit('graphUpdate', data.graph_data);
                break;
                
            case 'audio':
                this.handleAudioData(data.audio);
                break;
                
            case 'answer':
                this.emit('ragflowAnswer', data.answer);
                break;
                
            case 'error':
                this.emit('error', { type: 'server_error', message: data.message });
                break;
                
            case 'ragflowResponse':
                this.handleRagflowResponse(data);
                break;
                
            case 'openaiResponse':
                this.emit('openaiResponse', data.response);
                break;
                
            case 'simulation_mode_set':
                console.log('Simulation mode set:', data.mode);
                this.emit('simulationModeSet', data.mode);
                break;
                
            default:
                console.warn('Unhandled message type:', data.type);
                break;
        }
    }
}

export default WebsocketService;
