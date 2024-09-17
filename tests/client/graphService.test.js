// graphService.test.js

const GraphDataManager = require('../../public/js/services/graphDataManager');
const WebSocket = require('ws');

describe('GraphDataManager', () => {
  let graphDataManager;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = new WebSocket('ws://localhost:8443');
    graphDataManager = new GraphDataManager(mockWebSocket);
  });

  afterEach(() => {
    mockWebSocket.close();
    global.fetch.mockClear();
    delete global.fetch;
  });

  test('should initialize properly', () => {
    expect(graphDataManager).toBeDefined();
    expect(graphDataManager.websocket).toBeDefined();
  });

  test('should fetch initial graph data', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        json: () => Promise.resolve({ nodes: [], edges: [] }),
      })
    );

    const data = await graphDataManager.fetchInitialGraphData();

    expect(global.fetch).toHaveBeenCalledWith('/graph-data');
    expect(data).toEqual({ nodes: [], edges: [] });
  });

  test('should handle WebSocket messages', () => {
    const mockMessage = {
      data: JSON.stringify({
        type: 'nodePositions',
        positions: [{ id: 'node1', x: 1, y: 2, z: 3 }],
      }),
    };

    const onNodePositionsUpdateSpy = jest.fn();
    graphDataManager.onNodePositionsUpdate = onNodePositionsUpdateSpy;

    mockWebSocket.onmessage(mockMessage);

    expect(onNodePositionsUpdateSpy).toHaveBeenCalledWith([
      { id: 'node1', x: 1, y: 2, z: 3 },
    ]);
  });

  test('should send messages via WebSocket', () => {
    const sendSpy = jest.spyOn(mockWebSocket, 'send');

    graphDataManager.sendMessage({ type: 'startSimulation' });

    expect(sendSpy).toHaveBeenCalledWith(
      JSON.stringify({ type: 'startSimulation' })
    );
  });
});
