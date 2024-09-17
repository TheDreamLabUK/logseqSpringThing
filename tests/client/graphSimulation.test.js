// graphSimulation.test.js

const GraphSimulation = require('../../public/js/components/graphSimulation');

describe('GraphSimulation', () => {
  let graphSimulation;

  beforeEach(() => {
    graphSimulation = new GraphSimulation();
  });

  test('should initialize properly', () => {
    expect(graphSimulation).toBeDefined();
    expect(graphSimulation.nodes).toEqual([]);
    expect(graphSimulation.edges).toEqual([]);
  });

  test('should update node positions', () => {
    graphSimulation.nodes = [{ id: 'node1' }, { id: 'node2' }];

    const positions = [
      { id: 'node1', x: 1, y: 2, z: 3 },
      { id: 'node2', x: 4, y: 5, z: 6 },
    ];

    graphSimulation.updateNodePositions(positions);

    expect(graphSimulation.nodes[0]).toEqual({
      id: 'node1',
      x: 1,
      y: 2,
      z: 3,
    });
    expect(graphSimulation.nodes[1]).toEqual({
      id: 'node2',
      x: 4,
      y: 5,
      z: 6,
    });
  });

  test('should update data', () => {
    const newData = {
      nodes: [{ id: 'node3' }],
      edges: [{ source: 'node3', target: 'node1' }],
    };

    graphSimulation.updateData(newData);

    expect(graphSimulation.nodes).toEqual(newData.nodes);
    expect(graphSimulation.edges).toEqual(newData.edges);
  });
});
