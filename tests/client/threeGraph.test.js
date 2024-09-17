import { initThreeGraph, updateGraphVisuals, addNodeToGraph, addEdgeToGraph } from '../../public/js/threeJS/threeGraph';

describe('Three.js Graph', () => {
  let mockScene, graph;

  beforeEach(() => {
    mockScene = {};
    graph = initThreeGraph(mockScene);
  });

  test('initThreeGraph function exists', () => {
    expect(typeof initThreeGraph).toBe('function');
  });

  test('initThreeGraph returns an object with update, addNode, and addEdge methods', () => {
    expect(typeof graph.update).toBe('function');
    expect(typeof graph.addNode).toBe('function');
    expect(typeof graph.addEdge).toBe('function');
  });

  test('updateGraphVisuals function exists', () => {
    expect(typeof updateGraphVisuals).toBe('function');
  });

  test('updateGraphVisuals returns true', () => {
    expect(updateGraphVisuals(graph, {})).toBe(true);
  });

  test('addNodeToGraph function exists', () => {
    expect(typeof addNodeToGraph).toBe('function');
  });

  test('addNodeToGraph returns true', () => {
    expect(addNodeToGraph(graph, { id: 1 })).toBe(true);
  });

  test('addEdgeToGraph function exists', () => {
    expect(typeof addEdgeToGraph).toBe('function');
  });

  test('addEdgeToGraph returns true', () => {
    expect(addEdgeToGraph(graph, { source: 1, target: 2 })).toBe(true);
  });
});
