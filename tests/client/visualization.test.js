import { initVisualization, updateVisualization, resizeVisualization } from '../../public/js/components/visualization';

describe('Visualization', () => {
  test('initVisualization function exists', () => {
    expect(typeof initVisualization).toBe('function');
  });

  test('initVisualization function can be called without errors', () => {
    expect(() => initVisualization()).not.toThrow();
  });

  test('updateVisualization function exists', () => {
    expect(typeof updateVisualization).toBe('function');
  });

  test('updateVisualization function returns true', () => {
    const data = { nodes: [], edges: [] };
    expect(updateVisualization(data)).toBe(true);
  });

  test('resizeVisualization function exists', () => {
    expect(typeof resizeVisualization).toBe('function');
  });

  test('resizeVisualization function returns true', () => {
    expect(resizeVisualization(800, 600)).toBe(true);
  });
});
