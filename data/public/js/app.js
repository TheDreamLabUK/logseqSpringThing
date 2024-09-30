// public/js/app.js

import { WebsocketService } from './services/websocketService.js';
import { GraphDataManager } from './services/graphDataManager.js';
import { WebXRVisualization } from './components/webXRVisualization.js';
import { ChatManager } from './components/chatManager.js';
import { Interface } from './components/interface.js';
import { RAGflowService } from './services/ragflowService.js';

class App {
    constructor() {
        console.log('Initializing App');
        // Initialize services and components
        this.websocketService = new WebsocketService();
        this.graphDataManager = new GraphDataManager(this.websocketService);
        this.visualization = new WebXRVisualization(this.graphDataManager);
        this.chatManager = new ChatManager(this.websocketService);
        this.interface = new Interface(document);

        this.ragflowService = new RAGflowService(this.websocketService);

        this.initializeEventListeners();
    }

    /**
     * Sets up event listeners for WebSocket events and custom events.
     */
    initializeEventListeners() {
        console.log('Setting up event listeners');
        // WebSocket open event
        this.websocketService.on('open', () => {
            console.log('WebSocket connection established');
            this.graphDataManager.requestInitialData();
        });

        // WebSocket message event
        this.websocketService.on('message', (data) => {
            console.log('Received WebSocket message:', JSON.stringify(data, null, 2));
            if (data.type === 'graphUpdate') {
                console.log('Received graph update');
                this.graphDataManager.updateGraphData(data.graphData);
                // Visualization update is now handled by the 'graphDataUpdated' event
            } else if (data.type === 'chatResponse') {
                console.log('Received chat response:', data.message);
                this.chatManager.displayResponse(data.message);
            } else {
                console.warn('Unhandled message type:', data.type);
            }
        });

        // WebSocket error event
        this.websocketService.on('error', (error) => {
            console.error('WebSocket error:', error);
            this.interface.displayErrorMessage('WebSocket connection error.');
        });

        // WebSocket close event
        this.websocketService.on('close', () => {
            console.log('WebSocket connection closed');
            this.interface.displayErrorMessage('WebSocket connection closed.');
        });

        // Custom event for RAGFlow answers
        window.addEventListener('ragflowAnswer', (event) => {
            console.log('Received RAGFlow answer:', event.detail);
            const answer = event.detail;
            this.chatManager.displayResponse(answer);
        });

        // Fullscreen button event
        const fullscreenButton = document.getElementById('fullscreen-button');
        if (fullscreenButton) {
            fullscreenButton.addEventListener('click', this.toggleFullscreen.bind(this));
        } else {
            console.warn('Fullscreen button not found');
        }

        // Listen for graph data updates
        window.addEventListener('graphDataUpdated', (event) => {
            console.log('Graph data updated event received');
            this.visualization.updateVisualization();
        });
    }

    /**
     * Toggles the browser's fullscreen mode.
     */
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch((err) => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
                this.interface.displayErrorMessage('Unable to enter fullscreen mode.');
            });
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    }

    /**
     * Starts the application by initializing visualization and chat components.
     */
    start() {
        console.log('Starting the application');
        this.visualization.initialize();
        this.chatManager.initialize();
    }
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded, creating App instance');
    const app = new App();
    app.start();
});
