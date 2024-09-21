import { WebsocketService } from './services/websocketService.js';
import { GraphDataManager } from './services/graphDataManager.js';
import { WebXRVisualization } from './components/webXRVisualization.js';
import { ChatManager } from './components/chatManager.js';

class App {
    constructor() {
        this.websocketService = new WebsocketService();
        this.graphDataManager = new GraphDataManager(this.websocketService);
        this.webXRVisualization = new WebXRVisualization(this.graphDataManager);
        this.chatManager = new ChatManager(this.websocketService);

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        this.websocketService.on('open', () => {
            console.log('WebSocket connection established');
            this.graphDataManager.requestInitialData();
        });

        this.websocketService.on('message', (data) => {
            if (data.type === 'graphUpdate') {
                this.graphDataManager.updateGraphData(data.graphData);
                this.webXRVisualization.updateVisualization();
            } else if (data.type === 'chatResponse') {
                this.chatManager.displayResponse(data.message);
            }
        });

        this.websocketService.on('error', (error) => {
            console.error('WebSocket error:', error);
        });

        this.websocketService.on('close', () => {
            console.log('WebSocket connection closed');
        });
    }

    start() {
        this.webXRVisualization.initialize();
        this.chatManager.initialize();
    }
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.start();
});
