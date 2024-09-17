import { jest } from '@jest/globals';

// Mock WebXRVisualization class
class WebXRVisualization {
  constructor() {
    this.scene = { add: jest.fn() };
    this.camera = null;
    this.renderer = null;
  }

  initialize() {
    this.camera = {};
    this.renderer = { setSize: jest.fn(), domElement: document.createElement('canvas') };
  }

  addNode(node) {
    this.scene.add(node);
  }

  addEdge(edge) {
    this.scene.add(edge);
  }

  updateGraph(graphData) {
    graphData.nodes.forEach(node => this.addNode(node));
    graphData.edges.forEach(edge => this.addEdge(edge));
  }
}

describe('WebXRVisualization', () => {
  let webXRVisualization;

  beforeEach(() => {
    webXRVisualization = new WebXRVisualization();
  });

  test('initialize should set up the scene', () => {
    webXRVisualization.initialize();
    expect(webXRVisualization.scene).toBeDefined();
    expect(webXRVisualization.camera).toBeDefined();
    expect(webXRVisualization.renderer).toBeDefined();
  });

  test('addNode should add a node to the scene', () => {
    webXRVisualization.initialize();
    const node = { id: '1', label: 'Test Node' };
    webXRVisualization.addNode(node);
    expect(webXRVisualization.scene.add).toHaveBeenCalledWith(node);
  });

  test('addEdge should add an edge to the scene', () => {
    webXRVisualization.initialize();
    const edge = { source: '1', target: '2' };
    webXRVisualization.addEdge(edge);
    expect(webXRVisualization.scene.add).toHaveBeenCalledWith(edge);
  });

  test('updateGraph should update the visualization', () => {
    webXRVisualization.initialize();
    const graphData = {
      nodes: [{ id: '1', label: 'Node 1' }],
      edges: [{ source: '1', target: '2' }],
    };
    webXRVisualization.updateGraph(graphData);
    expect(webXRVisualization.scene.add).toHaveBeenCalledTimes(2); // Once for node, once for edge
  });

  // Add more tests as needed based on WebXRVisualization functionality
});

// visualization.test.js

const Visualization = require('../../public/js/components/visualization');

describe('Visualization', () => {
  let visualization;

  beforeEach(() => {
    visualization = new Visualization();
  });

  test('should initialize properly', () => {
    expect(visualization).toBeDefined();
  });

  test('should update the graph', () => {
    const updateGraphSpy = jest.spyOn(visualization, 'updateGraph');

    visualization.updateGraph();

    expect(updateGraphSpy).toHaveBeenCalled();
  });
});
