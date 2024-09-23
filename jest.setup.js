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
global.document = {
  createElement: jest.fn(() => ({})),  // Mock behavior as needed
};


// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock Three.js modules that are problematic in Jest
jest.mock('three/examples/jsm/controls/OrbitControls', () => {
  return {
    OrbitControls: jest.fn().mockImplementation(() => ({
      update: jest.fn(),
      dispose: jest.fn(),
    })),
  };
});

jest.mock('three/examples/jsm/misc/GPUComputationRenderer', () => {
  return {
    GPUComputationRenderer: jest.fn().mockImplementation(() => ({
      init: jest.fn(),
      compute: jest.fn(),
      getCurrentRenderTarget: jest.fn(),
      createTexture: jest.fn(),
    })),
  };
});


jest.mock('three/examples/jsm/webxr/VRButton', () => ({
  VRButton: {
    createButton: jest.fn(),
  },
}));

// Mock other components related to WebXR if needed
jest.mock('three/examples/jsm/webxr/XRControllerModelFactory', () => {
  return {
    XRControllerModelFactory: jest.fn().mockImplementation(() => ({
      createControllerModel: jest.fn(),
    })),
  };
});

// Mock 3d-force-graph
jest.mock('3d-force-graph', () => {
  return {
    ForceGraph3D: jest.fn().mockImplementation(() => ({
      graphData: jest.fn(),
      nodeColor: jest.fn(),
      linkColor: jest.fn(),
      nodeThreeObject: jest.fn(),
      onNodeClick: jest.fn(),
      onLinkClick: jest.fn(),
    })),
  };
});
>>>>>>> 5e408e194b8851565f2b1e0bb7217b0722c0bb15
