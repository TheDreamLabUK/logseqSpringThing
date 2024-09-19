// public/js/services/graphDataManager.js

/**
 * GraphDataManager handles fetching initial graph data and managing real-time updates.
 */
export class GraphDataManager {
  /**
   * Creates a new GraphDataManager instance.
   * @param {WebsocketService} webSocket - The WebSocket service instance.
   */
  constructor(webSocket) {
    this.webSocket = webSocket;
    this.graphData = { nodes: [], edges: [] };
  }

  /**
   * Fetches initial graph data from the server via HTTP.
   * @returns {Promise<void>}
   */
  async fetchInitialData() {
    try {
      const response = await fetch('/api/graph-data');
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      this.graphData = await response.json();
      console.log('Initial graph data loaded:', this.graphData);
    } catch (error) {
      console.error('Error loading graph data:', error);
      throw new Error('Error loading graph data');
    }
  }

  /**
   * Subscribes to real-time graph updates via WebSocket.
   * @param {function} callback - The function to call when new data is received.
   */
  subscribeToUpdates(callback) {
    this.webSocket.on('message', (data) => {
      if (data.type === 'graphUpdate') {
        this.graphData = data.data;
        console.log('Received graph update:', this.graphData);
        callback(this.graphData);
      }
    });
  }

  /**
   * Sends a message to the server via WebSocket.
   * @param {object} message - The message to send.
   */
  sendWebSocketMessage(message) {
    this.webSocket.send(message);
  }

  /**
   * Adds a new node to the graph.
   * @param {object} node - The node to add.
   */
  addNode(node) {
    this.graphData.nodes.push(node);
  }

  /**
   * Removes a node and its associated edges from the graph.
   * @param {string} nodeId - The ID of the node to remove.
   */
  removeNode(nodeId) {
    this.graphData.nodes = this.graphData.nodes.filter(node => node.id !== nodeId);
    this.graphData.edges = this.graphData.edges.filter(edge => edge.source !== nodeId && edge.target !== nodeId);
  }

  /**
   * Adds a new edge to the graph.
   * @param {object} edge - The edge to add.
   */
  addEdge(edge) {
    this.graphData.edges.push(edge);
  }

  /**
   * Removes an edge from the graph.
   * @param {string} source - The source node ID.
   * @param {string} target - The target node ID.
   */
  removeEdge(source, target) {
    this.graphData.edges = this.graphData.edges.filter(edge => !(edge.source === source && edge.target === target));
  }

  /**
   * Updates a node's properties.
   * @param {string} nodeId - The ID of the node to update.
   * @param {object} updates - The properties to update.
   */
  updateNode(nodeId, updates) {
    const nodeIndex = this.graphData.nodes.findIndex(node => node.id === nodeId);
    if (nodeIndex !== -1) {
      this.graphData.nodes[nodeIndex] = { ...this.graphData.nodes[nodeIndex], ...updates };
    }
  }

  /**
   * Retrieves a node by its ID.
   * @param {string} nodeId - The ID of the node to retrieve.
   * @returns {object|null} The node object or null if not found.
   */
  getNodeById(nodeId) {
    return this.graphData.nodes.find(node => node.id === nodeId) || null;
  }

  /**
   * Retrieves all nodes connected to a given node.
   * @param {string} nodeId - The ID of the node.
   * @returns {Array} An array of connected node objects.
   */
  getConnectedNodes(nodeId) {
    const connectedNodeIds = new Set();
    this.graphData.edges.forEach(edge => {
      if (edge.source === nodeId) connectedNodeIds.add(edge.target);
      if (edge.target === nodeId) connectedNodeIds.add(edge.source);
    });
    return this.graphData.nodes.filter(node => connectedNodeIds.has(node.id));
  }

  /**
   * Clears all graph data.
   */
  clearData() {
    this.graphData = { nodes: [], edges: [] };
  }
}
