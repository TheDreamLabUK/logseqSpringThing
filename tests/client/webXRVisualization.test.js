// webXRVisualization.test.js

const WebXRVisualization = require('../../public/js/components/webXRVisualization');
const THREE = require('three');

describe('WebXRVisualization', () => {
  let webXRVisualization;

  beforeEach(() => {
    webXRVisualization = new WebXRVisualization();
  });

  test('should initialize properly', () => {
    expect(webXRVisualization).toBeDefined();
    expect(webXRVisualization.renderer).toBeInstanceOf(THREE.WebGLRenderer);
    expect(webXRVisualization.scene).toBeInstanceOf(THREE.Scene);
    expect(webXRVisualization.camera).toBeInstanceOf(THREE.PerspectiveCamera);
  });

  test('should start XR session', () => {
    webXRVisualization.renderer.xr = { enabled: false };
    webXRVisualization.startXRSession();
    expect(webXRVisualization.renderer.xr.enabled).toBe(true);
  });

  test('should render the scene', () => {
    const renderSpy = jest.spyOn(webXRVisualization.renderer, 'render');

    webXRVisualization.render();

    expect(renderSpy).toHaveBeenCalledWith(
      webXRVisualization.scene,
      webXRVisualization.camera
    );
  });
});
