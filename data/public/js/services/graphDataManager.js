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
    }

    /**
     * Requests the initial graph data from the server via WebSocket.
     */
    requestInitialData() {
        this.websocketService.send({ type: 'getInitialData' });
    }

    /**
     * Updates the internal graph data with new data received from the server.
     * @param {object} newData - The new graph data.
     */
    updateGraphData(newData) {
        this.graphData = newData;
        console.log('Graph data updated:', this.graphData);
    }

    /**
     * Retrieves the current graph data.
     * @returns {object|null} The current graph data or null if not set.
     */
    getGraphData() {
        return this.graphData;
    }

    // Additional methods can be added here for more complex data management
}
