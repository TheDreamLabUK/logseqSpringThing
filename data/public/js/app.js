// data/public/js/app.js

import { createApp } from 'vue';
import ControlPanel from './components/ControlPanel.vue';
import { WebXRVisualization } from './components/visualization/core.js';
import WebsocketService from './services/websocketService.js';
import { GraphDataManager } from './services/graphDataManager.js';
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
        
        // Add debug info to DOM
        const debugInfo = document.getElementById('debug-info');
        if (debugInfo) {
            debugInfo.innerHTML = '<div>App constructor called</div>';
        }
    }

    async start() {
        console.log('Starting application');
        const debugInfo = document.getElementById('debug-info');
        try {
            await this.initializeApp();
            console.log('Application started successfully');
            if (debugInfo) {
                debugInfo.innerHTML += '<div style="color: #28a745">Application started successfully</div>';
            }
        } catch (error) {
            console.error('Failed to start application:', error);
            if (debugInfo) {
                debugInfo.innerHTML += `<div style="color: #dc3545">Error: ${error.message}</div>`;
            }
            throw error;
        }
    }

    async initializeApp() {
        const debugInfo = document.getElementById('debug-info');
        
        // Step 1: Initialize WebSocket and wait for initial data
        try {
            if (debugInfo) debugInfo.innerHTML += '<div>Initializing WebSocket...</div>';
            this.websocketService = new WebsocketService();
            
            // Wait for connection and initial data
            await new Promise((resolve, reject) => {
                let isConnected = false;
                let hasData = false;
                
                const timeout = setTimeout(() => {
                    if (isConnected) {
                        // If connected but no data, continue anyway
                        console.warn('Connected but waiting for initial data - continuing initialization');
                        resolve();
                    } else {
                        reject(new Error('WebSocket initialization timeout'));
                    }
                }, 10000);

                const checkComplete = () => {
                    if (isConnected && hasData) {
                        clearTimeout(timeout);
                        resolve();
                    }
                };

                this.websocketService.on('open', () => {
                    console.log('WebSocket connected');
                    if (debugInfo) debugInfo.innerHTML += '<div style="color: #28a745">WebSocket connected</div>';
                    isConnected = true;
                    checkComplete();
                });

                this.websocketService.on('message', (data) => {
                    console.log('Received message:', data);
                    if (data.type === 'graphData' || data.type === 'graphUpdate') {
                        if (debugInfo) debugInfo.innerHTML += '<div>Initial graph data received</div>';
                        hasData = true;
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
            
            if (debugInfo) debugInfo.innerHTML += '<div style="color: #28a745">WebSocket initialized successfully</div>';
        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
            if (debugInfo) debugInfo.innerHTML += `<div style="color: #dc3545">WebSocket Error: ${error.message}</div>`;
            throw error;
        }

        // Step 2: Initialize GraphDataManager
        try {
            if (debugInfo) debugInfo.innerHTML += '<div>Initializing GraphDataManager...</div>';
            this.graphDataManager = new GraphDataManager(this.websocketService);
            if (debugInfo) debugInfo.innerHTML += '<div style="color: #28a745">GraphDataManager initialized</div>';
        } catch (error) {
            console.error('Failed to initialize GraphDataManager:', error);
            if (debugInfo) debugInfo.innerHTML += `<div style="color: #dc3545">GraphDataManager Error: ${error.message}</div>`;
            throw error;
        }

        // Step 3: Initialize Scene Container
        try {
            if (debugInfo) debugInfo.innerHTML += '<div>Setting up scene container...</div>';
            const container = document.getElementById('scene-container');
            if (!container) {
                throw new Error('Scene container not found in DOM');
            }
            if (debugInfo) debugInfo.innerHTML += '<div style="color: #28a745">Scene container ready</div>';
        } catch (error) {
            console.error('Scene container error:', error);
            if (debugInfo) debugInfo.innerHTML += `<div style="color: #dc3545">Scene Container Error: ${error.message}</div>`;
            throw error;
        }

        // Step 4: Initialize Visualization
        try {
            if (debugInfo) debugInfo.innerHTML += '<div>Initializing visualization...</div>';
            this.visualization = new WebXRVisualization(this.graphDataManager);
            
            // Set up visualization event listener for graph updates
            window.addEventListener('graphDataUpdated', (event) => {
                if (event.detail && this.visualization) {
                    this.visualization.updateVisualization(event.detail);
                }
            });
            
            if (debugInfo) debugInfo.innerHTML += '<div style="color: #28a745">Visualization initialized</div>';
        } catch (error) {
            console.error('Failed to initialize visualization:', error);
            if (debugInfo) debugInfo.innerHTML += `<div style="color: #dc3545">Visualization Error: ${error.message}</div>`;
            throw error;
        }

        // Step 5: Initialize GPU (if available)
        try {
            if (debugInfo) debugInfo.innerHTML += '<div>Checking GPU availability...</div>';
            this.gpuAvailable = await isGPUAvailable();
            if (this.gpuAvailable) {
                this.gpuUtils = await initGPU();
                if (debugInfo) debugInfo.innerHTML += '<div style="color: #28a745">GPU acceleration enabled</div>';
            } else {
                if (debugInfo) debugInfo.innerHTML += '<div style="color: #ffc107">GPU not available, using CPU</div>';
            }
        } catch (error) {
            console.warn('GPU initialization failed:', error);
            if (debugInfo) debugInfo.innerHTML += `<div style="color: #ffc107">GPU Error: ${error.message}</div>`;
            // Don't throw error for GPU initialization failure
        }

        // Step 6: Initialize Vue App
        try {
            if (debugInfo) debugInfo.innerHTML += '<div>Initializing Vue app...</div>';
            await this.initVueApp();
            if (debugInfo) debugInfo.innerHTML += '<div style="color: #28a745">Vue app initialized</div>';
        } catch (error) {
            console.error('Failed to initialize Vue app:', error);
            if (debugInfo) debugInfo.innerHTML += `<div style="color: #dc3545">Vue App Error: ${error.message}</div>`;
            throw error;
        }

        // Step 7: Setup Event Listeners
        this.setupEventListeners();
    }

    async initVueApp() {
        const appContainer = document.getElementById('app');
        if (!appContainer) {
            throw new Error("Could not find '#app' element");
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
                        if (self.visualization) {
                            self.visualization.updateSettings(change);
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
            const debugInfo = document.getElementById('debug-info');
            if (debugInfo) {
                debugInfo.innerHTML += `<div style="color: #dc3545">Vue Error: ${err.message}</div>`;
            }
        };

        app.mount('#app');
        this.vueApp = app;
    }

    setupEventListeners() {
        if (this.websocketService) {
            this.websocketService.on('open', () => {
                console.log('WebSocket connected');
                const status = document.getElementById('connection-status');
                if (status) {
                    status.textContent = 'Connected';
                    status.className = 'connected';
                }
            });

            this.websocketService.on('close', () => {
                console.log('WebSocket disconnected');
                const status = document.getElementById('connection-status');
                if (status) {
                    status.textContent = 'Disconnected';
                    status.className = 'disconnected';
                }
            });

            // Handle initial graph data
            this.websocketService.on('message', (data) => {
                if (data.type === 'graphData' && this.graphDataManager) {
                    this.graphDataManager.updateGraphData(data);
                }
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
