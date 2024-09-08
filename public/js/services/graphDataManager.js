export class GraphDataManager {
  constructor() {
    this.ws = null;
    this.data = null;
  }

  connectWebSocket(url) {
    this.ws = new WebSocket(url);
    this.ws.onmessage = (event) => {
      this.data = JSON.parse(event.data);
      console.log('Received data:', this.data);
    };
  }

  requestGraphData() {
    if (this.ws) {
      this.ws.send(JSON.stringify({ type: 'getGraphData' }));
    }
  }
}
