import * as THREE from 'three';

// Mock Three.js
jest.mock('three', () => {
  return {
    ...jest.requireActual('three'),
    WebGLRenderer: jest.fn().mockImplementation(() => ({
      setSize: jest.fn(),
      render: jest.fn(),
      shadowMap: {},
    })),
    Scene: jest.fn(),
    PerspectiveCamera: jest.fn(),
    Vector3: jest.fn(),
    Raycaster: jest.fn(),
    Object3D: jest.fn(),
  };
});

// Mock other Three.js modules
jest.mock('three/examples/jsm/controls/OrbitControls', () => ({
  OrbitControls: jest.fn(),
}));

jest.mock('three/examples/jsm/webxr/VRButton', () => ({
  VRButton: {
    createButton: jest.fn(),
  },
}));

jest.mock('three/examples/jsm/webxr/XRControllerModelFactory', () => ({
  XRControllerModelFactory: jest.fn(),
}));

// Mock 3d-force-graph
jest.mock('3d-force-graph', () => ({
  ForceGraph3D: jest.fn().mockImplementation(() => ({
    graphData: jest.fn(),
    onNodeClick: jest.fn(),
    onLinkClick: jest.fn(),
  })),
}));

// Mock browser globals
global.URL.createObjectURL = jest.fn();
global.WebSocket = jest.fn(() => ({
  send: jest.fn(),
  close: jest.fn(),
}));

// Mock WebGL context
class WebGLRenderingContext {}
global.WebGLRenderingContext = WebGLRenderingContext;

// Mock navigator.xr
global.navigator.xr = {
  isSessionSupported: jest.fn().mockResolvedValue(true),
  requestSession: jest.fn().mockResolvedValue({}),
};

// Mock HTMLCanvasElement
class HTMLCanvasElement {
  getContext() {
    return new WebGLRenderingContext();
  }
}
global.HTMLCanvasElement = HTMLCanvasElement;

// Mock requestAnimationFrame
global.requestAnimationFrame = jest.fn(callback => setTimeout(callback, 0));

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Add any other necessary mocks here