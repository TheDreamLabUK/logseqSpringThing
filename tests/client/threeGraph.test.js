import { initializeThreeGraph } from '../../public/js/threeJS/threeGraph';

// Declare jest directly
const jest = require('jest-mock');

describe('ThreeGraph', () => {
  test('initializeThreeGraph should initialize Three.js graph correctly', () => {
    console.log = jest.fn(); // Mock console.log
    initializeThreeGraph();
    expect(console.log).toHaveBeenCalledWith('Initializing Three.js graph...');
  });
});