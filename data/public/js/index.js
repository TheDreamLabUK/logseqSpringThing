import { createApp } from 'vue';
import WebsocketService from './services/websocketService.js';
import { WebXRVisualization } from './components/visualization/core.js';
import ControlPanel from './components/ControlPanel.vue';

class App {
    constructor() {
        // Initialize WebSocket service first
        this.websocketService = new WebsocketService();
        
        // Initialize visualization with the websocket service as the graph data manager
        this.visualization = new WebXRVisualization(this.websocketService);
        this.vueApp = null;

        // Bind the updateVisualization method
        this.handleGraphData = this.handleGraphData.bind(this);
    }

    handleGraphData(data) {
        console.log('Received graph data:', data);
        if (this.visualization && data.nodes && data.edges) {
            // Transform edges data to match expected format
            const transformedEdges = data.edges.map(edge => ({
                source: edge.source,
                target_node: edge.target_node,
                weight: edge.weight || 1
            }));

            // Update visualization with the transformed data
            this.visualization.updateVisualization({
                nodes: data.nodes,
                edges: transformedEdges
            });
        }
    }

    async start() {
        try {
            // Initialize WebSocket connection
            await this.websocketService.connect();
            
            // Initialize visualization
            const container = document.getElementById('scene-container');
            if (!container) {
                throw new Error("Could not find 'scene-container' element");
            }
            await this.visualization.initThreeJS(container);

            // Set up WebSocket event handlers
            this.websocketService.on('graphData', this.handleGraphData);
            
            // Create Vue application
            const websocketService = this.websocketService;
            const visualization = this.visualization;
            
            this.vueApp = createApp({
                components: {
                    ControlPanel
                },
                setup() {
                    return {
                        websocketService,
                        handleControlChange(change) {
                            console.log('Control changed:', change);
                            visualization.updateSettings(change);
                        }
                    };
                },
                template: `
                    <div class="app-wrapper">
                        <ControlPanel 
                            :websocket-service="websocketService"
                            @control-change="handleControlChange"
                        />
                    </div>
                `
            });

            // Mount the application
            const appContainer = document.getElementById('app');
            if (!appContainer) {
                throw new Error("Could not find 'app' element");
            }
            this.vueApp.mount(appContainer);

            // Update connection status
            this.websocketService.on('connect', () => {
                const statusEl = document.getElementById('connection-status');
                if (statusEl) {
                    statusEl.textContent = 'Connected';
                    statusEl.className = 'connected';
                }
            });

            this.websocketService.on('disconnect', () => {
                const statusEl = document.getElementById('connection-status');
                if (statusEl) {
                    statusEl.textContent = 'Disconnected';
                    statusEl.className = 'disconnected';
                }
            });

            // Handle window resize
            window.addEventListener('resize', () => {
                if (this.visualization) {
                    this.visualization.onWindowResize();
                }
            });

        } catch (error) {
            console.error('Failed to start application:', error);
            throw error;
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

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.start().catch(error => {
        console.error('Failed to start application:', error);
    });

    // Handle cleanup on page unload
    window.addEventListener('beforeunload', () => {
        app.stop();
    });
});

export { App };
