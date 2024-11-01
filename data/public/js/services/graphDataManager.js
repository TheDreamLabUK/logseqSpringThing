// public/js/services/graphDataManager.js

/**
 * GraphDataManager handles the management and updating of graph data received from the server.
 */
export class GraphDataManager {
    /**
     * Creates a new GraphDataManager instance.
     * @param {WebsocketService} websocketService - The WebSocket service instance.
     */
    constructor(websocketService) {
        this.websocketService = websocketService;
        this.graphData = null;
        this.forceDirectedParams = {
            iterations: 100,
            repulsion: 1.0,
            attraction: 0.01
        };
        console.log('GraphDataManager initialized');
        
        // Set up WebSocket graph update listener
        this.websocketService.on('graphUpdate', this.handleGraphUpdate.bind(this));
    }

    /**
     * Requests the initial graph data from the server via WebSocket.
     */
    requestInitialData() {
        console.log('Requesting initial graph data');
        this.websocketService.send({ type: 'get_initial_data' }); // Fixed to match server's expected format
    }

    /**
     * Handles incoming graph update messages.
     * @param {object} graphData - The received graph data.
     */
    handleGraphUpdate(graphData) {
        console.log('Processing graph update:', graphData);
        // Check if the data is nested under a 'data' property
        const data = graphData.data || graphData;
        this.updateGraphData(data);
    }

    /**
     * Updates the internal graph data with new data received from the server.
     * @param {object} newData - The new graph data.
     */
    updateGraphData(newData) {
        console.log('Updating graph data with:', JSON.stringify(newData, null, 2));
        if (newData && Array.isArray(newData.edges)) {
            // Convert edges array to nodes and edges format if needed
            const graphData = {
                nodes: Array.from(new Set([
                    ...newData.edges.map(e => e.source),
                    ...newData.edges.map(e => e.target_node)
                ])).map(id => ({ id, label: id })),
                edges: newData.edges.map(e => ({
                    source: e.source,
                    target: e.target_node,
                    weight: e.weight,
                    hyperlinks: e.hyperlinks
                }))
            };
            
            this.graphData = graphData;
            console.log(`Graph data updated: ${graphData.nodes.length} nodes, ${graphData.edges.length} edges`);
            
            // Log some sample data
            if (graphData.nodes.length > 0) {
                console.log('Sample node:', JSON.stringify(graphData.nodes[0], null, 2));
            }
            if (graphData.edges.length > 0) {
                console.log('Sample edge:', JSON.stringify(graphData.edges[0], null, 2));
            }
            
            // Dispatch an event to notify that the graph data has been updated
            window.dispatchEvent(new CustomEvent('graphDataUpdated', { detail: this.graphData }));
        } else {
            console.error('Received invalid graph data:', newData);
        }
    }

    /**
     * Retrieves the current graph data.
     * @returns {object|null} The current graph data or null if not set.
     */
    getGraphData() {
        if (this.graphData) {
            console.log(`Returning graph data: ${this.graphData.nodes.length} nodes, ${this.graphData.edges.length} edges`);
        } else {
            console.warn('Graph data is null');
        }
        return this.graphData;
    }

    /**
     * Checks if the graph data is valid.
     * @returns {boolean} True if the graph data is valid, false otherwise.
     */
    isGraphDataValid() {
        return this.graphData && Array.isArray(this.graphData.nodes) && Array.isArray(this.graphData.edges);
    }

    /**
     * Updates the force-directed graph parameters.
     * @param {string} name - The name of the parameter to update.
     * @param {number} value - The new value for the parameter.
     */
    updateForceDirectedParams(name, value) {
        console.log(`Updating force-directed parameter: ${name} = ${value}`);
        if (name in this.forceDirectedParams) {
            this.forceDirectedParams[name] = value;
            console.log('Force-directed parameters updated:', this.forceDirectedParams);
        } else {
            console.warn(`Unknown force-directed parameter: ${name}`);
        }
    }

    /**
     * Recalculates the graph layout using the current force-directed parameters.
     */
    recalculateLayout() {
        console.log('Recalculating graph layout with parameters:', this.forceDirectedParams);
        if (this.isGraphDataValid()) {
            // Send a message to the server to recalculate the layout
            this.websocketService.send({
                type: 'recalculate_layout', // Fixed to match server's expected format
                params: this.forceDirectedParams
            });
            console.log('Layout recalculation requested');
            
            // Dispatch an event to notify that a layout recalculation has been requested
            window.dispatchEvent(new CustomEvent('layoutRecalculationRequested', {
                detail: this.forceDirectedParams
            }));
        } else {
            console.error('Cannot recalculate layout: Invalid graph data');
        }
    }
}
