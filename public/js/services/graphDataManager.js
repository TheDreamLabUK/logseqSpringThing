export class GraphDataManager {
  constructor(webSocket) {
    this.webSocket = webSocket;
    this.graphData = { nodes: [], edges: [] };
  }

  async fetchInitialData() {
    try {
      const response = await fetch('/api/graph-data');
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      this.graphData = await response.json();
    } catch (error) {
      console.error('Error loading graph data:', error);
      throw new Error('Error loading graph data');
    }
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

  addNode(node) {
    this.graphData.nodes.push(node);
  }

  removeNode(nodeId) {
    this.graphData.nodes = this.graphData.nodes.filter(node => node.id !== nodeId);
    this.graphData.edges = this.graphData.edges.filter(edge => edge.source !== nodeId && edge.target !== nodeId);
  }

  addEdge(edge) {
    this.graphData.edges.push(edge);
  }

  removeEdge(source, target) {
    this.graphData.edges = this.graphData.edges.filter(edge => !(edge.source === source && edge.target === target));
  }

  updateNode(nodeId, updates) {
    const nodeIndex = this.graphData.nodes.findIndex(node => node.id === nodeId);
    if (nodeIndex !== -1) {
      this.graphData.nodes[nodeIndex] = { ...this.graphData.nodes[nodeIndex], ...updates };
    }
  }

  getNodeById(nodeId) {
    return this.graphData.nodes.find(node => node.id === nodeId);
  }

  getConnectedNodes(nodeId) {
    const connectedNodeIds = new Set();
    this.graphData.edges.forEach(edge => {
      if (edge.source === nodeId) connectedNodeIds.add(edge.target);
      if (edge.target === nodeId) connectedNodeIds.add(edge.source);
    });
    return this.graphData.nodes.filter(node => connectedNodeIds.has(node.id));
  }

  clearData() {
    this.graphData = { nodes: [], edges: [] };
  }
}
