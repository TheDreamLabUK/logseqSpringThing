import * as THREE from 'three';
import { ForceGraph } from '../../public/js/threeJS/threeGraph';

jest.mock('three', () => {
  return {
    Scene: jest.fn(),
    SphereGeometry: jest.fn(),
    MeshBasicMaterial: jest.fn(),
    Mesh: jest.fn().mockImplementation(() => ({
      position: { set: jest.fn() }
    })),
    BufferGeometry: jest.fn().mockImplementation(() => ({
      setFromPoints: jest.fn().mockReturnThis()
    })),
    LineBasicMaterial: jest.fn(),
    Line: jest.fn().mockImplementation(() => ({
      geometry: { setFromPoints: jest.fn() }
    })),
    Vector3: jest.fn().mockImplementation((x, y, z) => ({ x, y, z }))
  };
});

describe('ForceGraph', () => {
  let forceGraph;
  let mockScene;

  beforeEach(() => {
    mockScene = {
      add: jest.fn(),
      remove: jest.fn()
    };
    forceGraph = new ForceGraph(mockScene);
  });

  test('should initialize with empty graph', () => {
    expect(forceGraph.nodes).toEqual([]);
    expect(forceGraph.links).toEqual([]);
    expect(forceGraph.nodeObjects.size).toBe(0);
    expect(forceGraph.linkObjects.size).toBe(0);
  });

  test('should update graph with new data', () => {
    const mockData = {
      nodes: [{ id: 1, x: 0, y: 0, z: 0 }, { id: 2, x: 1, y: 1, z: 1 }],
      links: [{ source: 1, target: 2 }]
    };

    forceGraph.updateGraph(mockData);

    expect(forceGraph.nodes).toEqual(mockData.nodes);
    expect(forceGraph.links).toEqual(mockData.links);
    expect(mockScene.add).toHaveBeenCalledTimes(3); // 2 nodes + 1 link
  });

  test('should update existing nodes and links', () => {
    const initialData = {
      nodes: [{ id: 1, x: 0, y: 0, z: 0 }],
      links: []
    };
    forceGraph.updateGraph(initialData);

    const updatedData = {
      nodes: [{ id: 1, x: 1, y: 1, z: 1 }, { id: 2, x: 2, y: 2, z: 2 }],
      links: [{ source: 1, target: 2 }]
    };
    forceGraph.updateGraph(updatedData);

    expect(forceGraph.nodeObjects.size).toBe(2);
    expect(forceGraph.linkObjects.size).toBe(1);
    expect(mockScene.add).toHaveBeenCalledTimes(3); // 2 nodes + 1 link
  });

  test('should remove nodes and links that no longer exist', () => {
    const initialData = {
      nodes: [{ id: 1, x: 0, y: 0, z: 0 }, { id: 2, x: 1, y: 1, z: 1 }],
      links: [{ source: 1, target: 2 }]
    };
    forceGraph.updateGraph(initialData);

    const updatedData = {
      nodes: [{ id: 1, x: 0, y: 0, z: 0 }],
      links: []
    };
    forceGraph.updateGraph(updatedData);

    expect(forceGraph.nodeObjects.size).toBe(1);
    expect(forceGraph.linkObjects.size).toBe(0);
    expect(mockScene.remove).toHaveBeenCalledTimes(2); // 1 node + 1 link removed
  });
});
