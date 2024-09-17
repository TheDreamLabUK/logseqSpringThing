import { GraphDataManager } from '../../public/js/services/graphDataManager';

describe('Graph Data Manager', () => {
  let graphDataManager;
  let mockWebSocket;

  beforeEach(() => {
    mockWebSocket = { send: jest.fn() };
    graphDataManager = new GraphDataManager(mockWebSocket);
  });

  test('GraphDataManager can be instantiated', () => {
    expect(graphDataManager).toBeInstanceOf(GraphDataManager);
  });

  test('fetchInitialData method exists', () => {
    expect(typeof graphDataManager.fetchInitialData).toBe('function');
  });

  test('handleWebSocketMessage method exists', () => {
    expect(typeof graphDataManager.handleWebSocketMessage).toBe('function');
  });

  test('sendWebSocketMessage method exists', () => {
    expect(typeof graphDataManager.sendWebSocketMessage).toBe('function');
  });

  test('getGraphData method exists', () => {
    expect(typeof graphDataManager.getGraphData).toBe('function');
  });

  test('getGraphData returns an object with nodes and edges', () => {
    const result = graphDataManager.getGraphData();
    expect(result).toHaveProperty('nodes');
    expect(result).toHaveProperty('edges');
    expect(Array.isArray(result.nodes)).toBe(true);
    expect(Array.isArray(result.edges)).toBe(true);
  });
});