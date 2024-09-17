import { GraphDataManager } from '../../public/js/services/graphDataManager';

// Mock WebSocket
class MockWebSocket {
  constructor(url) {
    this.url = url;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
  }

  send(data) {
    // Mock send method
  }

  close() {
    // Mock close method
  }
}

describe('GraphDataManager', () => {
  let graphDataManager;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = new MockWebSocket('ws://localhost:8443');
    graphDataManager = new GraphDataManager(mockWebSocket);
  });

  test('should initialize properly', () => {
    expect(graphDataManager).toBeDefined();
  });

  test('should fetch initial graph data', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      json: jest.fn().mockResolvedValue({ nodes: [], edges: [] }),
    });

    await graphDataManager.fetchInitialData();

    expect(global.fetch).toHaveBeenCalledWith('/api/graph-data');
    expect(graphDataManager.graphData).toEqual({ nodes: [], edges: [] });
  });

  test('should handle WebSocket messages', () => {
    const mockMessage = { data: JSON.stringify({ type: 'update', data: { nodes: [{ id: 1 }], edges: [] } }) };
    graphDataManager.handleWebSocketMessage(mockMessage);

    expect(graphDataManager.graphData).toEqual({ nodes: [{ id: 1 }], edges: [] });
  });

  test('should send messages via WebSocket', () => {
    const mockSend = jest.spyOn(mockWebSocket, 'send');
    graphDataManager.sendWebSocketMessage({ type: 'request', data: {} });

    expect(mockSend).toHaveBeenCalledWith(JSON.stringify({ type: 'request', data: {} }));
  });
});
