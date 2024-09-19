import * as THREE from 'three';

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
if (typeof global.navigator === 'undefined') {
  global.navigator = {};
}
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

jest.mock('three', () => {
  const actualThree = jest.requireActual('three');
  return {
    ...actualThree,
    WebGLRenderer: jest.fn().mockImplementation(() => ({
      setSize: jest.fn(),
      render: jest.fn(),
      domElement: {
        style: {},
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        clientWidth: 800,
        clientHeight: 600,
      },
      // Include any other necessary properties or methods
    })),
    Scene: jest.fn(() => ({
      add: jest.fn(),
      remove: jest.fn(),
      // Include other methods if needed
    })),
    PerspectiveCamera: jest.fn(() => ({
      position: { z: 100 },
      updateProjectionMatrix: jest.fn(),
      // Include other methods if needed
    })),
    // Do not mock Vector3, Matrix4, Raycaster, Object3D, etc.
  };
});
