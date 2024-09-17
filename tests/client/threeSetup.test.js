import { initThreeScene, createThreeCamera, createThreeRenderer, updateSceneSize } from '../../public/js/threeJS/threeSetup';

describe('Three.js Setup', () => {
  test('initThreeScene function exists', () => {
    expect(typeof initThreeScene).toBe('function');
  });

  test('initThreeScene returns an object with scene, camera, and renderer', () => {
    const result = initThreeScene();
    expect(result).toHaveProperty('scene');
    expect(result).toHaveProperty('camera');
    expect(result).toHaveProperty('renderer');
  });

  test('createThreeCamera function exists', () => {
    expect(typeof createThreeCamera).toBe('function');
  });

  test('createThreeCamera returns an object', () => {
    expect(typeof createThreeCamera()).toBe('object');
  });

  test('createThreeRenderer function exists', () => {
    expect(typeof createThreeRenderer).toBe('function');
  });

  test('createThreeRenderer returns an object with setSize and render methods', () => {
    const renderer = createThreeRenderer();
    expect(typeof renderer.setSize).toBe('function');
    expect(typeof renderer.render).toBe('function');
  });

  test('updateSceneSize function exists', () => {
    expect(typeof updateSceneSize).toBe('function');
  });

  test('updateSceneSize returns true', () => {
    const mockRenderer = { setSize: jest.fn() };
    const mockCamera = {};
    expect(updateSceneSize(mockRenderer, mockCamera, 800, 600)).toBe(true);
    expect(mockRenderer.setSize).toHaveBeenCalledWith(800, 600);
  });
});