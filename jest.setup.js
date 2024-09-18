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

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));