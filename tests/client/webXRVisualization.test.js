import { WebXRVisualization } from '../../public/js/components/webXRVisualization';
import * as THREE from 'three';


jest.mock('three/examples/jsm/webxr/VRButton', () => ({
  VRButton: {
    createButton: jest.fn(() => {
      const button = {};
      return button;
    })
  }
}));

jest.mock('three/examples/jsm/webxr/XRControllerModelFactory', () => ({
  XRControllerModelFactory: jest.fn().mockImplementation(() => ({
    createControllerModel: jest.fn(),
  })),
}));

describe('WebXRVisualization', () => {
  let webXRVisualization;
  let mockScene;
  let mockRenderer;
  let mockCamera;
  let mockWebsocketService;

  beforeEach(() => {
    mockScene = {
      add: jest.fn(),
      remove: jest.fn()
    };
    mockRenderer = {
      xr: {
        enabled: false,
        getController: jest.fn(() => ({
          addEventListener: jest.fn(),
          add: jest.fn(),
          matrixWorld: new THREE.Matrix4(),
          position: new THREE.Vector3()
        })),
        getControllerGrip: jest.fn(() => ({
          add: jest.fn()
        }))
      },
      setAnimationLoop: jest.fn(),
      render: jest.fn()
    };
    mockCamera = {};
    mockWebsocketService = {
      on: jest.fn(),
      send: jest.fn()
    };

    webXRVisualization = new WebXRVisualization(mockScene, mockRenderer, mockCamera, mockWebsocketService);
  });

  test('WebXRVisualization initializes correctly', () => {
    expect(webXRVisualization.scene).toBe(mockScene);
    expect(webXRVisualization.renderer).toBe(mockRenderer);
    expect(webXRVisualization.camera).toBe(mockCamera);
    expect(webXRVisualization.websocketService).toBe(mockWebsocketService);
    expect(mockWebsocketService.on).toHaveBeenCalledTimes(2);
  });

  test('initVR enables XR and adds VR button', () => {
    document.body.appendChild = jest.fn();
    const { VRButton } = require('three/examples/jsm/webxr/VRButton');

    webXRVisualization.initVR();

    expect(mockRenderer.xr.enabled).toBe(true);
    expect(VRButton.createButton).toHaveBeenCalledWith(mockRenderer);
    expect(document.body.appendChild).toHaveBeenCalled();
  });

  test('initVRControllers sets up VR controllers', () => {
    const { XRControllerModelFactory } = require('three/examples/jsm/webxr/XRControllerModelFactory');

    webXRVisualization.initVRControllers();

    expect(mockRenderer.xr.getController).toHaveBeenCalledTimes(2);
    expect(mockRenderer.xr.getControllerGrip).toHaveBeenCalledTimes(2);
    expect(mockScene.add).toHaveBeenCalledTimes(4);
    expect(XRControllerModelFactory).toHaveBeenCalledTimes(1);
  });

  test('onSelectStart handles controller selection', () => {
    const mockIntersection = { object: new THREE.Mesh() };
    webXRVisualization.getIntersections = jest.fn().mockReturnValue([mockIntersection]);
    webXRVisualization.selectNode = jest.fn();

    const mockEvent = { target: {} };
    webXRVisualization.onSelectStart(mockEvent);

    expect(webXRVisualization.getIntersections).toHaveBeenCalled();
    expect(webXRVisualization.selectNode).toHaveBeenCalledWith(mockIntersection.object);
  });

  test('onSelectEnd handles controller deselection', () => {
    webXRVisualization.deselectNode = jest.fn();

    webXRVisualization.onSelectEnd();

    expect(webXRVisualization.deselectNode).toHaveBeenCalled();
  });

  test('getIntersections returns intersections with nodes', () => {
    const mockController = {
      matrixWorld: new THREE.Matrix4(),
      position: new THREE.Vector3(0, 0, 0), // Ensure this is a THREE.Vector3
    };
    webXRVisualization.nodes = new Map([['node1', new THREE.Mesh()]]);

    const intersections = webXRVisualization.getIntersections(mockController);

    expect(intersections).toEqual([]);
    expect(webXRVisualization.nodes.size).toBe(1);
  });

  test('updateNodePositions updates node positions', () => {
    const mockNode = { position: { set: jest.fn() } };
    webXRVisualization.nodes = new Map([['node1', mockNode]]);

    webXRVisualization.updateNodePositions([
      { id: 'node1', position: { x: 1, y: 2, z: 3 } }
    ]);

    expect(mockNode.position.set).toHaveBeenCalledWith(1, 2, 3);
  });

  test('updateGraph updates the graph structure', () => {
    const mockGraphData = {
      nodes: [
        { id: 'node1', x: 1, y: 2, z: 3 }
      ]
    };

    webXRVisualization.updateGraph(mockGraphData);

    expect(mockScene.add).toHaveBeenCalled();
    expect(webXRVisualization.nodes.size).toBe(1);
  });

  test('animate sets up animation loop', () => {
    webXRVisualization.animate();

    expect(mockRenderer.setAnimationLoop).toHaveBeenCalled();
    const animationCallback = mockRenderer.setAnimationLoop.mock.calls[0][0];
    animationCallback();
    expect(mockRenderer.render).toHaveBeenCalledWith(mockScene, mockCamera);
  });
});
