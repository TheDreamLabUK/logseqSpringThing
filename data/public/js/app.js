// data/public/js/app.js

import { createApp } from 'vue';
import ControlPanel from './components/ControlPanel.vue';
import ChatManager from './components/chatManager.vue';
import { WebXRVisualization } from './components/visualization/core.js';
import WebsocketService from './services/websocketService.js';
import { GraphDataManager } from './services/graphDataManager.js';
import { isGPUAvailable, initGPU } from './gpuUtils.js';
import { enableSpacemouse } from './services/spacemouse.js';

export class App {
    constructor() {
        console.log('App constructor called');
        this.websocketService = null;
        this.graphDataManager = null;
        this.visualization = null;
        this.gpuAvailable = false;
        this.gpuUtils = null;
        this.xrActive = false;
        this.vueApp = null;
        
        // Add debug info to DOM
        const debugInfo = document.getElementById('debug-info');
        if (debugInfo) {
            debugInfo.innerHTML += '<div>App constructor called</div>';
        }
    }

    async start() {
        console.log('Starting application');
        try {
            await this.initializeApp();
            console.log('Application started successfully');
        } catch (error) {
            console.error('Failed to start application:', error);
            throw error;
        }
    }

    async initializeApp() {
        console.log('Initializing Application - Step 1: Services');

        // Initialize Services
        try {
            // Create WebsocketService and wait for connection
            this.websocketService = new WebsocketService();
            
            // Modified connection promise to handle both successful messages and connection event
            await new Promise((resolve, reject) => {
                let messageReceived = false;
                const timeout = setTimeout(() => {
                    if (!messageReceived) {
                        reject(new Error('WebSocket connection timeout'));
                    }
                }, 5000);

                this.websocketService.on('message', () => {
                    messageReceived = true;
                    clearTimeout(timeout);
                    resolve();
                });

                this.websocketService.on('error', (error) => {
                    clearTimeout(timeout);
                    reject(error);
                });
            });
            
            console.log('WebsocketService connected successfully');
        } catch (error) {
            console.error('Failed to initialize WebsocketService:', error);
            throw error;
        }

        // Initialize GraphDataManager after websocket is connected
        try {
            this.graphDataManager = new GraphDataManager(this.websocketService);
            console.log('GraphDataManager initialized successfully');
        } catch (error) {
            console.error('Failed to initialize GraphDataManager:', error);
            throw error;
        }
        
        console.log('Initializing Application - Step 2: Visualization');
        try {
            // Add container check
            const container = document.getElementById('scene-container');
            if (!container) {
                console.error('Scene container not found, creating it');
                const newContainer = document.createElement('div');
                newContainer.id = 'scene-container';
                document.body.appendChild(newContainer);
            }

            this.visualization = new WebXRVisualization(this.graphDataManager);
            console.log('WebXRVisualization initialized successfully');
        } catch (error) {
            console.error('Failed to initialize WebXRVisualization:', error);
            console.error('Error stack:', error.stack);
            throw error;
        }

        console.log('Initializing Application - Step 3: GPU');
        // Initialize GPU if available
        this.gpuAvailable = await isGPUAvailable();
        if (this.gpuAvailable) {
            this.gpuUtils = await initGPU();
            console.log('GPU acceleration initialized');
        } else {
            console.warn('GPU acceleration not available, using CPU fallback');
        }

        console.log('Initializing Application - Step 4: Three.js');
        // Initialize Three.js first
        if (this.visualization) {
            await this.visualization.initThreeJS();
        } else {
            throw new Error('Visualization not initialized, cannot call initThreeJS');
        }

        console.log('Initializing Application - Step 5: Vue App');
        // Initialize Vue App with ChatManager and ControlPanel after Three.js
        await this.initVueApp();

        console.log('Initializing Application - Step 6: Event Listeners');
        // Setup Event Listeners
        this.setupEventListeners();

        // Request initial data after everything is initialized
        console.log('Requesting initial graph data');
        this.websocketService.send({ type: 'getInitialData' });
    }

    async initVueApp() {
        try {
            console.log('Initializing Vue application');
            
            const self = this; // Store reference to the App instance
            
            // Create Vue app with explicit template and debug styling
            const app = createApp({
                components: {
                    ControlPanel
                },
                template: `
                    <div style="position: fixed; top: 0; right: 0; z-index: 1000; background: rgba(0,0,0,0.8); padding: 20px;">
                        <control-panel 
                            :websocket-service="websocketService"
                            @control-change="handleControlChange"
                        />
                    </div>
                `,
                setup() {
                    return {
                        websocketService: self.websocketService,
                        handleControlChange: (change) => {
                            console.log('Control changed:', change);
                            self.visualization.updateSettings(change);
                        }
                    };
                }
            });

            // Add error handler
            app.config.errorHandler = (err, vm, info) => {
                console.error('Vue Error:', err);
                console.error('Error Info:', info);
            };

            // Mount with verification
            const appContainer = document.getElementById('app');
            if (!appContainer) {
                throw new Error("Could not find '#app' element");
            }

            app.mount('#app');
            this.vueApp = app;

            console.log('Vue application mounted successfully');
        } catch (error) {
            console.error('Failed to initialize Vue application:', error);
            throw error;
        }
    }

    setupEventListeners() {
        if (this.websocketService) {
            // Setup websocket event listeners
            this.websocketService.on('connect', () => {
                console.log('WebSocket connected');
                // Re-request initial data if reconnected
                this.websocketService.send({ type: 'getInitialData' });
            });

            this.websocketService.on('disconnect', () => {
                console.log('WebSocket disconnected');
            });

            // Add debug listener for graph updates
            this.websocketService.on('graphUpdate', (data) => {
                console.log('Received graph update:', data);
            });
        }
    }

    stop() {
        if (this.visualization) {
            this.visualization.dispose();
        }
        if (this.websocketService) {
            this.websocketService.disconnect();
        }
        if (this.vueApp) {
            this.vueApp.unmount();
        }
    }
}
