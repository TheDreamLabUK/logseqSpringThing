import { initWebXR } from '../../public/js/components/webXRVisualization';

// Declare jest directly
const jest = require('jest-mock');

describe('WebXRVisualization', () => {
  test('initWebXR should initialize WebXR visualization correctly', () => {
    console.log = jest.fn(); // Mock console.log
    initWebXR();
    expect(console.log).toHaveBeenCalledWith('Initializing WebXR visualization...');
  });
});