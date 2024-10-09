// data/public/js/app.js

import { createApp } from 'vue';
import { WebsocketService } from './services/websocketService.js';
import { GraphDataManager } from './services/graphDataManager.js';
import { WebXRVisualization } from './components/webXRVisualization.js';
import ChatManager from './components/chatManager.vue';
import { Interface } from './components/interface.js';
import { isGPUAvailable, initGPU } from './gpuUtils.js';
import { enableSpacemouse } from './spacemouse.js';

class App {
    constructor() {
        this.initializeApp();
    }

    initializeApp() {
        console.log('Initializing Application');

        // Initialize Services
        this.websocketService = new WebsocketService();
        this.graphDataManager = new GraphDataManager(this.websocketService);
        this.visualization = new WebXRVisualization(this.graphDataManager);
        this.interface = new Interface(document);

        // Initialize GPU if available
        this.gpuAvailable = isGPUAvailable();
        if (this.gpuAvailable) {
            this.gpuUtils = initGPU();
            console.log('GPU acceleration initialized');
        } else {
            console.warn('GPU acceleration not available, using CPU fallback');
        }

        // Initialize Vue App
        this.initVueApp();

        // Setup Event Listeners
        this.setupEventListeners();
    }

    initVueApp() {
        const app = createApp(ChatManager);
        app.config.globalProperties.$websocketService = this.websocketService;
        app.mount('#chat-container');
    }

    setupEventListeners() {
        console.log('Setting up event listeners');

        // WebSocket Event Listeners
        this.websocketService.on('open', () => {
            console.log('WebSocket connection established');
            this.updateConnectionStatus(true);
            this.graphDataManager.requestInitialData();
        });

        this.websocketService.on('message', (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        });

        this.websocketService.on('error', (error) => {
            console.error('WebSocket error:', error);
            this.interface.displayErrorMessage('WebSocket connection error.');
            this.updateConnectionStatus(false);
        });

        this.websocketService.on('close', () => {
            console.log('WebSocket connection closed');
            this.interface.displayErrorMessage('WebSocket connection closed.');
            this.updateConnectionStatus(false);
        });

        // Custom Event Listener for Graph Data Updates
        window.addEventListener('graphDataUpdated', (event) => {
            console.log('Graph data updated event received');
            this.visualization.updateVisualization();
        });

        // Fullscreen Button Event Listener
        const fullscreenButton = document.getElementById('fullscreen-button');
        if (fullscreenButton) {
            fullscreenButton.addEventListener('click', this.toggleFullscreen.bind(this));
        } else {
            console.warn('Fullscreen button not found');
        }

        // Spacemouse Button Event Listener
        const spacemouseButton = document.getElementById('enable-spacemouse');
        if (spacemouseButton) {
            spacemouseButton.addEventListener('click', enableSpacemouse);
        } else {
            console.warn('Spacemouse button not found');
        }

        // Spacemouse Move Event Listener
        window.addEventListener('spacemouse-move', (event) => {
            const { x, y, z } = event.detail;
            this.visualization.handleSpacemouseInput(x, y, z);
        });

        // Initialize audio on first user interaction
        const initAudioOnUserInteraction = () => {
            this.websocketService.initAudio();
            console.log('Audio initialized on user interaction');
            document.removeEventListener('click', initAudioOnUserInteraction);
            document.removeEventListener('touchstart', initAudioOnUserInteraction);
        };
        document.addEventListener('click', initAudioOnUserInteraction);
        document.addEventListener('touchstart', initAudioOnUserInteraction);
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'graphUpdate':
                this.graphDataManager.updateGraphData(data.graphData);
                this.visualization.updateVisualization();
                break;
            case 'ttsMethodSet':
                this.interface.updateTTSToggleState(data.method);
                break;
            // Handle additional message types here
            default:
                console.warn(`Unhandled message type: ${data.type}`);
                break;
        }
    }

    updateConnectionStatus(isConnected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = isConnected ? 'Connected' : 'Disconnected';
            statusElement.className = isConnected ? 'connected' : 'disconnected';
        } else {
            console.warn('Connection status element not found');
        }
    }

    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch((err) => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
                this.interface.displayErrorMessage('Unable to enter fullscreen mode.');
            });
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen().catch((err) => {
                    console.error(`Error attempting to exit fullscreen: ${err.message}`);
                });
            }
        }
    }

    start() {
        console.log('Starting the application');
        this.visualization.initialize();
        // ChatManager is initialized through Vue
        if (this.gpuAvailable) {
            console.log('GPU acceleration is available');
            // Implement GPU-accelerated features here if needed
        } else {
            console.log('GPU acceleration is not available, using CPU fallback');
        }
    }
}

// Initialize the App once the DOM content is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded, creating App instance');
    const app = new App();
    app.start();
});
