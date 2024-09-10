import { initializeThreeGraph } from '../../public/js/threeJS/threeGraph';

describe('ThreeGraph', () => {
  test('initializeThreeGraph should initialize Three.js graph correctly', () => {
    const consoleSpy = jest.spyOn(console, 'log');
    initializeThreeGraph();
    expect(consoleSpy).toHaveBeenCalledWith('Initializing Three.js graph...');
    consoleSpy.mockRestore();
  });
});