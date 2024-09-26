// public/js/app.js

import { WebsocketService } from './services/websocketService.js';
import { GraphDataManager } from './services/graphDataManager.js';
import { Visualization } from './components/visualization.js';
import { ChatManager } from './components/chatManager.js';
import { Interface } from './components/interface.js';
import { RAGflowService } from './services/ragflowService.js';

class App {
    constructor() {
        // Initialize services and components
        this.websocketService = new WebsocketService();
        this.graphDataManager = new GraphDataManager(this.websocketService);
        this.visualization = new Visualization(this.graphDataManager);
        this.chatManager = new ChatManager(this.websocketService);
        this.interface = new Interface(document);

        this.ragflowService = new RAGflowService(this.websocketService);

        this.initializeEventListeners();
    }

    /**
     * Sets up event listeners for WebSocket events and custom events.
     */
    initializeEventListeners() {
        // WebSocket open event
        this.websocketService.on('open', () => {
            console.log('WebSocket connection established');
            this.graphDataManager.requestInitialData();
        });

        // WebSocket message event
        this.websocketService.on('message', (data) => {
            if (data.type === 'graphUpdate') {
                this.graphDataManager.updateGraphData(data.graphData);
                this.visualization.updateVisualization();
            } else if (data.type === 'chatResponse') {
                this.chatManager.displayResponse(data.message);
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
            const answer = event.detail;
            this.chatManager.displayResponse(answer);
        });

        // Fullscreen button event
        const fullscreenButton = document.getElementById('fullscreen-button');
        fullscreenButton.addEventListener('click', this.toggleFullscreen.bind(this));
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
        this.visualization.initialize();
        this.chatManager.initialize();
    }
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.start();
});
