import { JSDOM } from 'jsdom';

// Create a simulated DOM using JSDOM
const { window } = new JSDOM('<!doctype html><html><body></body></html>');
global.window = window;
global.document = window.document;

jest.mock('three', () => {
    const originalThree = jest.requireActual('three'); // Use the actual 'three' module for non-mocked parts
    return {
      ...originalThree, // Spread the original module to retain non-mocked functionality
      Scene: jest.fn(() => ({
        add: jest.fn(),
      })),
      PerspectiveCamera: jest.fn(() => ({
        position: { z: 100 },
        updateProjectionMatrix: jest.fn(),
      })),
      WebGLRenderer: jest.fn(() => ({
        setSize: jest.fn(),
        render: jest.fn(),
        domElement: {
          // Simplified mock of a DOM element
          style: {},
          addEventListener: jest.fn(),
          removeEventListener: jest.fn(),
          clientWidth: 800,
          clientHeight: 600,
        }
      })),
      Line: jest.fn(() => ({
        name: '',
        scale: { z: 5 },
      })),
      Matrix4: jest.fn(() => ({
        identity: jest.fn().mockReturnThis(),
        extractRotation: jest.fn(),
      })),
      Raycaster: jest.fn(() => ({
        ray: {
          origin: { setFromMatrixPosition: jest.fn() },
        },
      })),
    };
  });
  
  jest.mock('three/examples/jsm/webxr/VRButton', () => ({
    createButton: jest.fn(),
  }));
  
  jest.mock('three/examples/jsm/webxr/XRControllerModelFactory', () => ({
    createControllerModel: jest.fn(),
  }));