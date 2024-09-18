import { Visualization } from '../../public/js/components/visualization';

jest.mock('three');
jest.mock('three/examples/jsm/controls/OrbitControls');

describe('Visualization', () => {
  let visualization;
  let mockRenderer;
  let mockScene;
  let mockCamera;
  let mockControls;

  beforeEach(() => {
    mockRenderer = {
      setSize: jest.fn(),
      render: jest.fn(),
      domElement: document.createElement('div'),
    };
    mockScene = new THREE.Scene();
    mockCamera = new THREE.PerspectiveCamera();
    mockControls = new OrbitControls();

    THREE.WebGLRenderer.mockImplementation(() => mockRenderer);
    THREE.Scene.mockImplementation(() => mockScene);
    THREE.PerspectiveCamera.mockImplementation(() => mockCamera);
    OrbitControls.mockImplementation(() => mockControls);

    document.body.appendChild = jest.fn();
    window.addEventListener = jest.fn();

    visualization = new Visualization();
  });

  test('Visualization initializes correctly', () => {
    expect(visualization.scene).toBeInstanceOf(THREE.Scene);
    expect(visualization.camera).toBeInstanceOf(THREE.PerspectiveCamera);
    expect(visualization.renderer).toBe(mockRenderer);
    expect(visualization.controls).toBe(mockControls);
  });

  test('initThreeJS sets up Three.js environment', () => {
    expect(THREE.Scene).toHaveBeenCalled();
    expect(THREE.PerspectiveCamera).toHaveBeenCalled();
    expect(THREE.WebGLRenderer).toHaveBeenCalled();
    expect(OrbitControls).toHaveBeenCalled();
    expect(mockRenderer.setSize).toHaveBeenCalledWith(window.innerWidth, window.innerHeight);
    expect(document.body.appendChild).toHaveBeenCalledWith(mockRenderer.domElement);
    expect(window.addEventListener).toHaveBeenCalledWith('resize', expect.any(Function), false);
  });

  test('createNodeObjects creates node meshes', () => {
    const mockNodes = [{ id: 1, x: 0, y: 0, z: 0 }, { id: 2, x: 1, y: 1, z: 1 }];
    visualization.createNodeObjects(mockNodes);
    expect(mockScene.add).toHaveBeenCalledWith(expect.any(THREE.Group));
  });

  test('createEdgeObjects creates edge lines', () => {
    const mockEdges = [{ source: { x: 0, y: 0, z: 0 }, target: { x: 1, y: 1, z: 1 } }];
    visualization.createEdgeObjects(mockEdges);
    expect(mockScene.add).toHaveBeenCalledWith(expect.any(THREE.Group));
  });

  test('onWindowResize updates camera and renderer', () => {
    visualization.onWindowResize();
    expect(mockCamera.updateProjectionMatrix).toHaveBeenCalled();
    expect(mockRenderer.setSize).toHaveBeenCalledWith(window.innerWidth, window.innerHeight);
  });

  test('animate updates controls and renders', () => {
    jest.spyOn(window, 'requestAnimationFrame').mockImplementation(cb => cb());
    visualization.animate();
    expect(mockControls.update).toHaveBeenCalled();
    expect(mockRenderer.render).toHaveBeenCalledWith(mockScene, mockCamera);
  });
});
