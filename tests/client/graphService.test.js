// Mock GraphDataManager class
class GraphDataManager {
  constructor() {
    this.graphData = { nodes: [], edges: [] };
  }

  updateGraphData(newData) {
    this.graphData = newData;
  }

  getGraphData() {
    return this.graphData;
  }

  addNode(node) {
    this.graphData.nodes.push(node);
  }

  addEdge(edge) {
    this.graphData.edges.push(edge);
  }

  removeNode(nodeId) {
    this.graphData.nodes = this.graphData.nodes.filter(node => node.id !== nodeId);
  }
}

describe('GraphDataManager', () => {
  let graphDataManager;

  beforeEach(() => {
    graphDataManager = new GraphDataManager();
  });

  test('updateGraphData should update the graph data', () => {
    const newData = { nodes: [{ id: '1', label: 'Node 1' }], edges: [{ source: '1', target: '2' }] };
    graphDataManager.updateGraphData(newData);
    expect(graphDataManager.getGraphData()).toEqual(newData);
  });

  test('addNode should add a new node to the graph', () => {
    const newNode = { id: '1', label: 'New Node' };
    graphDataManager.addNode(newNode);
    expect(graphDataManager.getGraphData().nodes).toContainEqual(newNode);
  });

  test('addEdge should add a new edge to the graph', () => {
    const newEdge = { source: '1', target: '2' };
    graphDataManager.addEdge(newEdge);
    expect(graphDataManager.getGraphData().edges).toContainEqual(newEdge);
  });

  test('removeNode should remove a node from the graph', () => {
    const node = { id: '1', label: 'Node to remove' };
    graphDataManager.addNode(node);
    graphDataManager.removeNode('1');
    expect(graphDataManager.getGraphData().nodes).not.toContainEqual(node);
  });

  // Add more tests as needed based on GraphDataManager functionality
});