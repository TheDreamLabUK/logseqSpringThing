import { initWebXR } from '../../public/js/components/webXRVisualization';

describe('WebXRVisualization', () => {
  test('initWebXR should initialize WebXR visualization correctly', () => {
    const consoleSpy = jest.spyOn(console, 'log');
    initWebXR();
    expect(consoleSpy).toHaveBeenCalledWith('Initializing WebXR visualization...');
    consoleSpy.mockRestore();
  });
});