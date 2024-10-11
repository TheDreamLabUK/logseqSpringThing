// data/public/js/app.js

import { createApp } from 'vue';
import ControlPanel from './components/ControlPanel.vue';
import ChatManager from './components/chatManager.vue';
import { WebXRVisualization } from './components/webXRVisualization.js';
import WebsocketService from './services/websocketService.js';
import { GraphDataManager } from './services/graphDataManager.js';
import { isGPUAvailable, initGPU } from './gpuUtils.js';
import { enableSpacemouse } from './services/spacemouse.js';

class App {
    constructor() {
        console.log('App constructor called');
        this.initializeApp();
    }

    initializeApp() {
        console.log('Initializing Application');

        // Initialize Services
        this.websocketService = new WebsocketService();
        this.graphDataManager = new GraphDataManager(this.websocketService);
        this.visualization = new WebXRVisualization(this.graphDataManager);

        // Initialize GPU if available
        this.gpuAvailable = isGPUAvailable();
        if (this.gpuAvailable) {
            this.gpuUtils = initGPU();
        } else {
            console.warn('GPU acceleration not available, using CPU fallback');
        // ... (keep existing GPU initialization code)
        // Initialize the visualization
        this.visualization.initThreeJS();
    }

    initVueApp() {
        const app = createApp({
            components: {
                ControlPanel,
                ChatManager
            },
        } else {
            console.error('Visualization not initialized, cannot call initThreeJS');
        }
    }

    // ... (keep existing initVueApp method)

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
            console.log('WebSocket message received:', event);
            this.visualization.handleSpacemouseInput(x, y, z);
        });

        // Initialize audio on first user interaction
        let audioContext;

        document.addEventListener('click', initAudio, { once: true });

        function initAudio() {
        // ... (keep other existing WebSocket listeners)
        switch (data.type) {
            case 'graphUpdate':
                console.log('Received graph update:', data.graphData);
                this.graphDataManager.updateGraphData(data.graphData);
                this.visualization.updateVisualization();
                break;
            case 'ttsMethodSet':
                // Handle TTS method set
                break;
            // Handle additional message types here
            default:
        // ... (keep other existing event listeners)
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
        console.log('Handling WebSocket message:', data);
            if (document.exitFullscreen) {
                document.exitFullscreen().catch((err) => {
                    console.error(`Error attempting to exit fullscreen: ${err.message}`);
                });
            }
        }
    }

    start() {
        console.log('Starting the application');
            // ... (keep other cases)
            // Implement GPU-accelerated features here if needed
        } else {
            console.log('GPU acceleration is not available, using CPU fallback');
        }
    }
}
    // ... (keep other existing methods)
