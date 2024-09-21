export class GraphDataManager {
    constructor(websocketService) {
        this.websocketService = websocketService;
        this.graphData = null;
    }

    requestInitialData() {
        this.websocketService.send({ type: 'requestInitialData' });
    }

    updateGraphData(newData) {
        this.graphData = newData;
    }

    getGraphData() {
        return this.graphData;
    }

    requestGraphUpdate() {
        this.websocketService.send({ type: 'requestGraphUpdate' });
    }
}
