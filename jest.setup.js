// Mock browser globals
global.THREE = require('three');
global.WebSocket = jest.fn();

// Mock WebGL context
const mockWebGLContext = {
  getExtension: jest.fn(),
  createBuffer: jest.fn(),
  bindBuffer: jest.fn(),
  bufferData: jest.fn(),
  // Add more WebGL methods as needed
};

HTMLCanvasElement.prototype.getContext = jest.fn(() => mockWebGLContext);

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
