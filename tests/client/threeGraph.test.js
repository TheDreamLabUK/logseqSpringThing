// threeGraph.test.js

const ThreeGraph = require('../../public/js/threeJS/threeGraph');
const THREE = require('three');

describe('ThreeGraph', () => {
  let threeGraph;

  beforeEach(() => {
    threeGraph = new ThreeGraph();
  });

  test('should initialize properly', () => {
    expect(threeGraph).toBeDefined();
    expect(threeGraph.scene).toBeInstanceOf(THREE.Scene);
    expect(threeGraph.nodesGroup).toBeInstanceOf(THREE.Group);
    expect(threeGraph.edgesGroup).toBeInstanceOf(THREE.Group);
  });

  test('should add nodes to the scene', () => {
    const nodes = [{ id: 'node1', x: 1, y: 2, z: 3 }];

    threeGraph.addNodes(nodes);

    expect(threeGraph.nodesGroup.children.length).toBe(1);
    expect(threeGraph.nodesGroup.children[0]).toBeInstanceOf(THREE.Mesh);
  });

  test('should add edges to the scene', () => {
    const edges = [{ source: 'node1', target: 'node2' }];

    threeGraph.nodes = {
      node1: new THREE.Object3D(),
      node2: new THREE.Object3D(),
    };

    threeGraph.addEdges(edges);

    expect(threeGraph.edgesGroup.children.length).toBe(1);
    expect(threeGraph.edgesGroup.children[0]).toBeInstanceOf(THREE.Line);
  });
});
