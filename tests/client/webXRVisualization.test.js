import { WebXRVisualization } from '../../public/js/components/webXRVisualization';

jest.mock('three');
jest.mock('three/examples/jsm/webxr/VRButton');
jest.mock('three/examples/jsm/webxr/XRControllerModelFactory');

describe('WebXRVisualization', () => {
  let webXRVisualization;
  let mockScene;
  let mockRenderer;
  let mockCamera;

  beforeEach(() => {
    mockScene = new THREE.Scene();
    mockRenderer = {
      xr: {
        enabled: false,
        getController: jest.fn(),
        getControllerGrip: jest.fn()
      },
      setAnimationLoop: jest.fn()
    };
    mockCamera = new THREE.PerspectiveCamera();

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
    const mockController = new THREE.Object3D();
    const mockControllerGrip = new THREE.Object3D();
    mockRenderer.xr.getController.mockReturnValue(mockController);
    mockRenderer.xr.getControllerGrip.mockReturnValue(mockControllerGrip);

    webXRVisualization.initVRControllers();

    expect(mockRenderer.xr.getController).toHaveBeenCalledTimes(2);
    expect(mockRenderer.xr.getControllerGrip).toHaveBeenCalledTimes(2);
    expect(mockScene.add).toHaveBeenCalledTimes(4); // 2 controllers + 2 grips
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
    const mockRaycaster = {
      ray: {
        origin: new THREE.Vector3(),
        direction: new THREE.Vector3()
      },
      intersectObject: jest.fn().mockReturnValue(['mockIntersection'])
    };
    THREE.Raycaster.mockImplementation(() => mockRaycaster);

    webXRVisualization.nodesMesh = new THREE.Object3D();

    const intersections = webXRVisualization.getIntersections(mockController);

    expect(intersections).toEqual(['mockIntersection']);
    expect(mockRaycaster.intersectObject).toHaveBeenCalledWith(webXRVisualization.nodesMesh);
  });

  test('animate sets up animation loop', () => {
    const mockRender = jest.fn();
    webXRVisualization.render = mockRender;

    webXRVisualization.animate();

    expect(mockRenderer.setAnimationLoop).toHaveBeenCalled();
    const animationCallback = mockRenderer.setAnimationLoop.mock.calls[0][0];
    animationCallback();
    expect(mockRender).toHaveBeenCalled();
  });
});
