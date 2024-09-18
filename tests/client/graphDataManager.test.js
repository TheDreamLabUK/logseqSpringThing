import { GraphDataManager } from '../../public/js/services/graphDataManager';

describe('GraphDataManager', () => {
  let graphDataManager;

  beforeEach(() => {
    graphDataManager = new GraphDataManager();
  });

  test('GraphDataManager initializes correctly', () => {
    expect(graphDataManager.nodes).toEqual([]);
    expect(graphDataManager.edges).toEqual([]);
  });

  test('loadInitialData fetches and sets graph data', async () => {
    const mockData = {
      nodes: [{ id: 1, name: 'Node 1' }, { id: 2, name: 'Node 2' }],
      edges: [{ source: 1, target: 2 }]
    };

    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: jest.fn().mockResolvedValue(mockData)
    });

    await graphDataManager.loadInitialData();

    expect(global.fetch).toHaveBeenCalledWith('/api/graph-data');
    expect(graphDataManager.nodes).toEqual(mockData.nodes);
    expect(graphDataManager.edges).toEqual(mockData.edges);
  });

  test('loadInitialData handles fetch error', async () => {
    global.fetch = jest.fn().mockRejectedValue(new Error('Network error'));

    await expect(graphDataManager.loadInitialData()).rejects.toThrow('Error loading graph data');
  });

  test('addNode adds a new node', () => {
    const newNode = { id: 1, name: 'New Node' };
    graphDataManager.addNode(newNode);

    expect(graphDataManager.nodes).toContain(newNode);
  });

  test('removeNode removes a node and its associated edges', () => {
    graphDataManager.nodes = [{ id: 1, name: 'Node 1' }, { id: 2, name: 'Node 2' }];
    graphDataManager.edges = [{ source: 1, target: 2 }, { source: 2, target: 1 }];

    graphDataManager.removeNode(1);

    expect(graphDataManager.nodes).toEqual([{ id: 2, name: 'Node 2' }]);
    expect(graphDataManager.edges).toEqual([]);
  });

  test('addEdge adds a new edge', () => {
    const newEdge = { source: 1, target: 2 };
    graphDataManager.addEdge(newEdge);

    expect(graphDataManager.edges).toContain(newEdge);
  });

  test('removeEdge removes an edge', () => {
    graphDataManager.edges = [{ source: 1, target: 2 }, { source: 2, target: 3 }];

    graphDataManager.removeEdge(1, 2);

    expect(graphDataManager.edges).toEqual([{ source: 2, target: 3 }]);
  });

  test('updateNode updates an existing node', () => {
    graphDataManager.nodes = [{ id: 1, name: 'Old Name' }];

    graphDataManager.updateNode(1, { name: 'New Name' });

    expect(graphDataManager.nodes[0]).toEqual({ id: 1, name: 'New Name' });
  });

  test('getNodeById returns the correct node', () => {
    graphDataManager.nodes = [{ id: 1, name: 'Node 1' }, { id: 2, name: 'Node 2' }];

    const node = graphDataManager.getNodeById(2);

    expect(node).toEqual({ id: 2, name: 'Node 2' });
  });

  test('getConnectedNodes returns nodes connected to a given node', () => {
    graphDataManager.nodes = [{ id: 1 }, { id: 2 }, { id: 3 }];
    graphDataManager.edges = [{ source: 1, target: 2 }, { source: 2, target: 3 }];

    const connectedNodes = graphDataManager.getConnectedNodes(2);

    expect(connectedNodes).toEqual([{ id: 1 }, { id: 3 }]);
  });

  test('clearData clears all nodes and edges', () => {
    graphDataManager.nodes = [{ id: 1 }, { id: 2 }];
    graphDataManager.edges = [{ source: 1, target: 2 }];

    graphDataManager.clearData();

    expect(graphDataManager.nodes).toEqual([]);
    expect(graphDataManager.edges).toEqual([]);
  });
});