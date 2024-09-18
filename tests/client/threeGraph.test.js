import { ForceGraph } from '../../public/js/threeJS/threeGraph';

jest.mock('three');

describe('ForceGraph', () => {
  let forceGraph;
  let mockScene;

  beforeEach(() => {
    mockScene = new THREE.Scene();
    forceGraph = new ForceGraph(mockScene);
  });

  test('should initialize with an empty graph', () => {
    expect(forceGraph.nodes).toEqual([]);
    expect(forceGraph.links).toEqual([]);
  });

  test('should add nodes to the graph', () => {
    forceGraph.addNode({ id: 1 });
    forceGraph.addNode({ id: 2 });
    expect(forceGraph.nodes.length).toBe(2);
  });

  test('should add links to the graph', () => {
    forceGraph.addNode({ id: 1 });
    forceGraph.addNode({ id: 2 });
    forceGraph.addLink({ source: 1, target: 2 });
    expect(forceGraph.links.length).toBe(1);
  });

  test('should update node positions', () => {
    const node = { id: 1 };
    forceGraph.addNode(node);
    forceGraph.updateNodePosition(node, { x: 1, y: 2, z: 3 });
    expect(node.x).toBe(1);
    expect(node.y).toBe(2);
    expect(node.z).toBe(3);
  });

  test('should render the graph', () => {
    const mockRender = jest.fn();
    forceGraph.render = mockRender;
    forceGraph.renderGraph();
    expect(mockRender).toHaveBeenCalled();
  });
});
