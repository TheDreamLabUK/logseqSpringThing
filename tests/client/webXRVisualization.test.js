import { WebXRVisualization } from '../../public/js/components/webXRVisualization';

jest.mock('three', () => {
  const actualThree = jest.requireActual('three');
  return {
    ...actualThree,
    BufferGeometry: jest.fn().mockImplementation(() => ({
      setFromPoints: jest.fn().mockReturnThis()
    })),
    Vector3: jest.fn().mockImplementation(() => ({
      set: jest.fn().mockReturnThis(),
      applyMatrix4: jest.fn().mockReturnThis()
    })),
    Line: jest.fn().mockImplementation(() => ({
      name: '',
      scale: { z: 0 }
    })),
    Matrix4: jest.fn().mockImplementation(() => ({
      identity: jest.fn().mockReturnThis(),
      extractRotation: jest.fn().mockReturnThis()
    })),
    Raycaster: jest.fn().mockImplementation(() => ({
      ray: {
        origin: { setFromMatrixPosition: jest.fn() },
        direction: { set: jest.fn().mockReturnThis(), applyMatrix4: jest.fn() }
      },
      intersectObject: jest.fn().mockReturnValue([])
    }))
  };
});
jest.mock('three/examples/jsm/webxr/VRButton');
jest.mock('three/examples/jsm/webxr/XRControllerModelFactory');

describe('WebXRVisualization', () => {
  let webXRVisualization;
  let mockScene;
  let mockRenderer;
  let mockCamera;

  beforeEach(() => {
    mockScene = {
      add: jest.fn()
    };
    mockRenderer = {
      xr: {
        enabled: false,
        getController: jest.fn(() => ({
          addEventListener: jest.fn(),
          add: jest.fn()
        })),
        getControllerGrip: jest.fn(() => ({
          add: jest.fn()
        }))
      },
      setAnimationLoop: jest.fn(),
      render: jest.fn()
    };
    mockCamera = {};

    webXRVisualization = new WebXRVisualization(mockScene, mockRenderer, mockCamera);
  });

  test('WebXRVisualization initializes correctly', () => {
    expect(webXRVisualization.scene).toBe(mockScene);
    expect(webXRVisualization.renderer).toBe(mockRenderer);
    expect(webXRVisualization.camera).toBe(mockCamera);
  });

  test('initVR enables XR and adds VR button', () => {
    document.body.appendChild = jest.fn();
    VRButton.createButton.mockReturnValue(document.createElement('button'));

    webXRVisualization.initVR();

    expect(mockRenderer.xr.enabled).toBe(true);
    expect(VRButton.createButton).toHaveBeenCalledWith(mockRenderer);
    expect(document.body.appendChild).toHaveBeenCalled();
  });

  test('initVRControllers sets up VR controllers', () => {
    const mockControllerModelFactory = {
      createControllerModel: jest.fn()
    };
    XRControllerModelFactory.mockImplementation(() => mockControllerModelFactory);

    webXRVisualization.initVRControllers();

    expect(mockRenderer.xr.getController).toHaveBeenCalledTimes(2);
    expect(mockRenderer.xr.getControllerGrip).toHaveBeenCalledTimes(2);
    expect(mockScene.add).toHaveBeenCalledTimes(4);
    expect(XRControllerModelFactory).toHaveBeenCalledTimes(1);
  });

  test('onSelectStart handles controller selection', () => {
    const mockIntersection = { point: new THREE.Vector3() };
    webXRVisualization.getIntersections = jest.fn().mockReturnValue([mockIntersection]);
    webXRVisualization.selectNode = jest.fn();

    const mockEvent = { target: {} };
    webXRVisualization.onSelectStart(mockEvent);

    expect(webXRVisualization.getIntersections).toHaveBeenCalled();
    expect(webXRVisualization.selectNode).toHaveBeenCalledWith(mockIntersection.point);
  });

  test('onSelectEnd handles controller deselection', () => {
    webXRVisualization.deselectNode = jest.fn();

    webXRVisualization.onSelectEnd();

    expect(webXRVisualization.deselectNode).toHaveBeenCalled();
  });

  test('getIntersections returns intersections with nodes', () => {
    const mockController = {
      matrixWorld: new THREE.Matrix4()
    };
    webXRVisualization.nodesMesh = {};

    const intersections = webXRVisualization.getIntersections(mockController);

    expect(intersections).toEqual([]);
    expect(THREE.Raycaster).toHaveBeenCalled();
  });

  test('animate sets up animation loop', () => {
    webXRVisualization.animate();

    expect(mockRenderer.setAnimationLoop).toHaveBeenCalled();
    const animationCallback = mockRenderer.setAnimationLoop.mock.calls[0][0];
    animationCallback();
    expect(mockRenderer.render).toHaveBeenCalledWith(mockScene, mockCamera);
  });
});
