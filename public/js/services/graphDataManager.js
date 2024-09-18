export class GraphDataManager {
  constructor() {
    this.nodes = [];
    this.edges = [];
  }

  async loadInitialData() {
    try {
      const response = await fetch('/api/graph-data');
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      this.nodes = data.nodes;
      this.edges = data.edges;
    } catch (error) {
      console.error('Error loading graph data:', error);
      throw new Error('Error loading graph data');
    }
  }

  addNode(node) {
    this.nodes.push(node);
  }

  removeNode(nodeId) {
    this.nodes = this.nodes.filter(node => node.id !== nodeId);
    this.edges = this.edges.filter(edge => edge.source !== nodeId && edge.target !== nodeId);
  }

  addEdge(edge) {
    this.edges.push(edge);
  }

  removeEdge(source, target) {
    this.edges = this.edges.filter(edge => !(edge.source === source && edge.target === target));
  }

  updateNode(nodeId, updates) {
    const nodeIndex = this.nodes.findIndex(node => node.id === nodeId);
    if (nodeIndex !== -1) {
      this.nodes[nodeIndex] = { ...this.nodes[nodeIndex], ...updates };
    }
  }

  getNodeById(nodeId) {
    return this.nodes.find(node => node.id === nodeId);
  }

  getConnectedNodes(nodeId) {
    const connectedNodeIds = new Set();
    this.edges.forEach(edge => {
      if (edge.source === nodeId) connectedNodeIds.add(edge.target);
      if (edge.target === nodeId) connectedNodeIds.add(edge.source);
    });
    return this.nodes.filter(node => connectedNodeIds.has(node.id));
  }

  clearData() {
    this.nodes = [];
    this.edges = [];
  }
}
