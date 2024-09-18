import { Visualization } from '../../public/js/components/visualization';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';

jest.mock('three');
jest.mock('three/examples/jsm/controls/OrbitControls');
jest.mock('three/examples/jsm/postprocessing/EffectComposer');
jest.mock('three/examples/jsm/postprocessing/RenderPass');
jest.mock('three/examples/jsm/postprocessing/UnrealBloomPass');

describe('Visualization', () => {
  let visualization;
  let mockRenderer;
  let mockScene;
  let mockCamera;
  let mockControls;
  let mockComposer;

  beforeEach(() => {
    mockRenderer = {
      setSize: jest.fn(),
      render: jest.fn(),
      domElement: document.createElement('div'),
    };
    mockScene = new THREE.Scene();
    mockCamera = new THREE.PerspectiveCamera();
    mockControls = new OrbitControls();
    mockComposer = {
      addPass: jest.fn(),
      setSize: jest.fn(),
      render: jest.fn(),
    };

    THREE.WebGLRenderer.mockImplementation(() => mockRenderer);
    THREE.Scene.mockImplementation(() => mockScene);
    THREE.PerspectiveCamera.mockImplementation(() => mockCamera);
    OrbitControls.mockImplementation(() => mockControls);
    EffectComposer.mockImplementation(() => mockComposer);

    document.body.appendChild = jest.fn();
    window.addEventListener = jest.fn();

    visualization = new Visualization();
  });

  test('Visualization initializes correctly', () => {
    expect(visualization.scene).toBeInstanceOf(THREE.Scene);
    expect(visualization.camera).toBeInstanceOf(THREE.PerspectiveCamera);
    expect(visualization.renderer).toBe(mockRenderer);
    expect(visualization.controls).toBe(mockControls);
    expect(visualization.composer).toBe(mockComposer);
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

  test('addLights adds lights to the scene', () => {
    expect(mockScene.add).toHaveBeenCalledTimes(3); // Ambient, Point, and Directional lights
  });

  test('addBackground sets scene background', () => {
    expect(THREE.CubeTextureLoader).toHaveBeenCalled();
    expect(mockScene.background).toBeDefined();
  });

  test('initPostProcessing sets up post-processing', () => {
    expect(EffectComposer).toHaveBeenCalledWith(mockRenderer);
    expect(RenderPass).toHaveBeenCalledWith(mockScene, mockCamera);
    expect(UnrealBloomPass).toHaveBeenCalled();
    expect(mockComposer.addPass).toHaveBeenCalledTimes(2);
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
    expect(mockComposer.setSize).toHaveBeenCalledWith(window.innerWidth, window.innerHeight);
  });

  test('animate updates controls and renders', () => {
    jest.spyOn(window, 'requestAnimationFrame').mockImplementation(cb => cb());
    visualization.animate();
    expect(mockControls.update).toHaveBeenCalled();
    expect(mockComposer.render).toHaveBeenCalled();
  });
});
