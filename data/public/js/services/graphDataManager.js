export class GraphDataManager {
    constructor(websocketService) {
        this.websocketService = websocketService;
        this.graphData = null;
    }

    requestInitialData() {
        this.websocketService.send({ type: 'getInitialData' });
    }

    updateGraphData(newData) {
        this.graphData = newData;
        console.log('Graph data updated:', this.graphData);
    }

    // Add more methods as needed for graph data management
}
