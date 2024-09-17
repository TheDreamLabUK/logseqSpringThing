import { initGraphSimulation, updateGraphPositions } from '../../public/js/components/graphSimulation';

describe('Graph Simulation', () => {
  test('initGraphSimulation function exists', () => {
    expect(typeof initGraphSimulation).toBe('function');
  });

  test('initGraphSimulation function can be called without errors', () => {
    expect(() => initGraphSimulation()).not.toThrow();
  });

  test('updateGraphPositions function exists', () => {
    expect(typeof updateGraphPositions).toBe('function');
  });

  test('updateGraphPositions function returns the input nodes', () => {
    const nodes = [{ id: 1 }, { id: 2 }];
    expect(updateGraphPositions(nodes)).toBe(nodes);
  });
});
