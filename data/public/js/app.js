// data/public/js/app.js

import { createApp } from 'vue';
import ControlPanel from './components/ControlPanel.vue';
import { WebXRVisualization } from './components/visualization/core.js';
import WebsocketService from './services/websocketService.js';
import { GraphDataManager } from './services/graphDataManager.js';
import { visualizationSettings } from './services/visualizationSettings.js';
import { isGPUAvailable, initGPU } from './gpuUtils.js';

export class App {
    constructor() {
        console.log('App constructor called');
        this.websocketService = null;
        this.graphDataManager = null;
        this.visualization = null;
        this.gpuAvailable = false;
        this.gpuUtils = null;
        this.vueApp = null;
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
        // Step 1: Initialize WebSocket and wait for initial data and settings
        try {
            this.websocketService = new WebsocketService();
            
            // Wait for connection, initial data, and settings
            await new Promise((resolve, reject) => {
                let isConnected = false;
                let hasData = false;
                let hasSettings = false;
                
                const timeout = setTimeout(() => {
                    reject(new Error('Initialization timeout'));
                }, 10000);

                const checkComplete = () => {
                    if (isConnected && hasData && hasSettings) {
                        clearTimeout(timeout);
                        resolve();
                    }
                };

                this.websocketService.on('open', () => {
                    console.log('WebSocket connected');
                    isConnected = true;
                    checkComplete();
                });

                this.websocketService.on('message', (data) => {
                    console.log('Received message:', data);
                    if (data.type === 'graphData' || data.type === 'graphUpdate') {
                        hasData = true;
                        checkComplete();
                    }
                });

                this.websocketService.on('serverSettings', (settings) => {
                    console.log('Received server settings:', settings);
                    if (settings?.visualization) {
                        hasSettings = true;
                        checkComplete();
                    }
                });

                this.websocketService.on('error', (error) => {
                    if (!isConnected) {
                        clearTimeout(timeout);
                        reject(error);
                    } else {
                        console.error('WebSocket error after connection:', error);
                    }
                });
            });
        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
            throw error;
        }

        // Step 2: Initialize GraphDataManager
        try {
            this.graphDataManager = new GraphDataManager(this.websocketService);
        } catch (error) {
            console.error('Failed to initialize GraphDataManager:', error);
            throw error;
        }

        // Step 3: Initialize Scene Container
        try {
            const container = document.getElementById('scene-container');
            if (!container) {
                const sceneContainer = document.createElement('div');
                sceneContainer.id = 'scene-container';
                document.body.appendChild(sceneContainer);
                console.log('Created scene container');
            }
        } catch (error) {
            console.error('Scene container error:', error);
            throw error;
        }

        // Step 4: Verify Settings
        const settings = visualizationSettings.getSettings();
        if (!settings?.visualization) {
            throw new Error('Visualization settings not available');
        }

        // Step 5: Initialize Visualization
        try {
            // Wait for initial graph data to be processed
            await new Promise(resolve => setTimeout(resolve, 100));
            
            this.visualization = new WebXRVisualization(this.graphDataManager);
        } catch (error) {
            console.error('Failed to initialize visualization:', error);
            throw error;
        }

        // Step 6: Initialize GPU (if available)
        try {
            this.gpuAvailable = await isGPUAvailable();
            if (this.gpuAvailable) {
                this.gpuUtils = await initGPU();
            }
        } catch (error) {
            console.warn('GPU initialization failed:', error);
            // Don't throw error for GPU initialization failure
        }

        // Step 7: Initialize Vue App
        try {
            await this.initVueApp();
        } catch (error) {
            console.error('Failed to initialize Vue app:', error);
            throw error;
        }

        // Step 8: Setup Event Listeners
        this.setupEventListeners();
    }

    async initVueApp() {
        const appContainer = document.getElementById('app');
        if (!appContainer) {
            const app = document.createElement('div');
            app.id = 'app';
            document.body.appendChild(app);
            console.log('Created app container');
        }

        const self = this;
        
        const app = createApp({
            components: {
                ControlPanel
            },
            template: `
                <div class="control-panel-container">
                    <control-panel 
                        :websocket-service="websocketService"
                        @control-change="handleControlChange"
                    />
                </div>
            `,
            setup() {
                if (!self.websocketService) {
                    throw new Error('WebsocketService not initialized');
                }
                
                return {
                    websocketService: self.websocketService,
                    handleControlChange: (change) => {
                        console.log('Control changed:', change);
                        // Send settings update to server
                        if (self.websocketService) {
                            self.websocketService.updateSettings(change);
                        }
                    }
                };
            }
        });

        // Add global styles for Vue app
        const style = document.createElement('style');
        style.textContent = `
            .control-panel-container {
                position: fixed;
                top: 0;
                right: 0;
                z-index: 1000;
                background: rgba(0, 0, 0, 0.8);
                padding: 20px;
                pointer-events: auto;
            }
        `;
        document.head.appendChild(style);

        app.config.errorHandler = (err, vm, info) => {
            console.error('Vue Error:', err);
            console.error('Error Info:', info);
        };

        app.mount('#app');
        this.vueApp = app;
    }

    setupEventListeners() {
        if (this.websocketService) {
            this.websocketService.on('open', () => {
                console.log('WebSocket connected');
            });

            this.websocketService.on('close', () => {
                console.log('WebSocket disconnected');
            });

            // Listen for settings updates
            this.websocketService.on('serverSettings', (settings) => {
                console.log('Received settings update:', settings);
                // Settings will be automatically propagated through visualizationSettings service
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
