import { initWebXR } from '../../public/js/components/webXRVisualization';

describe('WebXR Visualization', () => {
  test('initWebXR function exists', () => {
    expect(typeof initWebXR).toBe('function');
  });

  test('initWebXR function can be called without errors', () => {
    expect(() => initWebXR()).not.toThrow();
  });
});
