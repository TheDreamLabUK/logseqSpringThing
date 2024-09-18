import { GraphDataManager } from '../../public/js/services/graphDataManager';

describe('GraphDataManager', () => {
  let graphDataManager;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = {
      send: jest.fn()
    };
    graphDataManager = new GraphDataManager(mockWebSocket);
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  test('should initialize with empty graph data', () => {
    expect(graphDataManager.graphData).toEqual({ nodes: [], edges: [] });
  });

  test('should fetch initial graph data', async () => {
    const mockData = { nodes: [{ id: 1 }], edges: [{ source: 1, target: 2 }] };
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: jest.fn().mockResolvedValueOnce(mockData)
    });

    await graphDataManager.fetchInitialData();

    expect(global.fetch).toHaveBeenCalledWith('/api/graph-data');
    expect(graphDataManager.graphData).toEqual(mockData);
  });

  test('should handle fetch error', async () => {
    global.fetch.mockRejectedValueOnce(new Error('Network error'));

    await expect(graphDataManager.fetchInitialData()).rejects.toThrow('Error loading graph data');
  });

  test('should handle WebSocket messages', () => {
    const mockMessage = { data: JSON.stringify({ type: 'update', data: { nodes: [{ id: 1 }], edges: [] } }) };
    graphDataManager.handleWebSocketMessage(mockMessage);

    expect(graphDataManager.graphData).toEqual({ nodes: [{ id: 1 }], edges: [] });
  });

  test('should send messages via WebSocket', () => {
    const message = { type: 'request', data: {} };
    graphDataManager.sendWebSocketMessage(message);

    expect(mockWebSocket.send).toHaveBeenCalledWith(JSON.stringify(message));
  });

  test('should add a node', () => {
    const node = { id: 1, label: 'Test Node' };
    graphDataManager.addNode(node);

    expect(graphDataManager.graphData.nodes).toContainEqual(node);
  });

  test('should remove a node', () => {
    graphDataManager.graphData = {
      nodes: [{ id: 1 }, { id: 2 }],
      edges: [{ source: 1, target: 2 }]
    };

    graphDataManager.removeNode(1);

    expect(graphDataManager.graphData.nodes).toEqual([{ id: 2 }]);
    expect(graphDataManager.graphData.edges).toEqual([]);
  });

  test('should add an edge', () => {
    const edge = { source: 1, target: 2 };
    graphDataManager.addEdge(edge);

    expect(graphDataManager.graphData.edges).toContainEqual(edge);
  });

  test('should remove an edge', () => {
    graphDataManager.graphData = {
      nodes: [{ id: 1 }, { id: 2 }],
      edges: [{ source: 1, target: 2 }, { source: 2, target: 1 }]
    };

    graphDataManager.removeEdge(1, 2);

    expect(graphDataManager.graphData.edges).toEqual([{ source: 2, target: 1 }]);
  });

  test('should update a node', () => {
    graphDataManager.graphData = {
      nodes: [{ id: 1, label: 'Old Label' }],
      edges: []
    };

    graphDataManager.updateNode(1, { label: 'New Label' });

    expect(graphDataManager.graphData.nodes[0]).toEqual({ id: 1, label: 'New Label' });
  });

  test('should get a node by id', () => {
    graphDataManager.graphData = {
      nodes: [{ id: 1, label: 'Test Node' }],
      edges: []
    };

    const node = graphDataManager.getNodeById(1);

    expect(node).toEqual({ id: 1, label: 'Test Node' });
  });

  test('should get connected nodes', () => {
    graphDataManager.graphData = {
      nodes: [{ id: 1 }, { id: 2 }, { id: 3 }],
      edges: [{ source: 1, target: 2 }, { source: 3, target: 1 }]
    };

    const connectedNodes = graphDataManager.getConnectedNodes(1);

    expect(connectedNodes).toEqual([{ id: 2 }, { id: 3 }]);
  });

  test('should clear data', () => {
    graphDataManager.graphData = {
      nodes: [{ id: 1 }],
      edges: [{ source: 1, target: 2 }]
    };

    graphDataManager.clearData();

    expect(graphDataManager.graphData).toEqual({ nodes: [], edges: [] });
  });
});
