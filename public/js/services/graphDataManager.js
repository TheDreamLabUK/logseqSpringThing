export class GraphDataManager {
  constructor(webSocket) {
    this.webSocket = webSocket;
    this.graphData = { nodes: [], edges: [] };
    console.log('Initializing graph data manager...');
  }

  async fetchInitialData() {
    const response = await fetch('/api/graph-data');
    this.graphData = await response.json();
    return this.graphData;
  }

  handleWebSocketMessage(message) {
    const data = JSON.parse(message.data);
    if (data.type === 'update') {
      this.graphData = data.data;
    }
  }

  sendWebSocketMessage(message) {
    this.webSocket.send(JSON.stringify(message));
  }

  getGraphData() {
    console.log('Getting graph data');
    return this.graphData;
  }
}
