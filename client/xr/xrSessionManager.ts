/**
 * XR session management and rendering
 */

import * as THREE from 'three';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory';
import { createLogger } from '../core/utils';
import { platformManager } from '../platform/platformManager';
import { SceneManager } from '../rendering/scene';
import { BACKGROUND_COLOR } from '../core/constants';

const _logger = createLogger('XRSessionManager');

// Type guards for WebXR features
function hasLightEstimate(frame: XRFrame): frame is XRFrame & { getLightEstimate(): XRLightEstimate | null } {
  return 'getLightEstimate' in frame;
}

function hasHitTest(session: XRSession): session is XRSession & { requestHitTestSource(options: XRHitTestOptionsInit): Promise<XRHitTestSource> } {
  return 'requestHitTestSource' in session;
}

export class XRSessionManager {
  private static instance: XRSessionManager;
  private sceneManager: SceneManager;
  private session: XRSession | null = null;
  private referenceSpace: XRReferenceSpace | null = null;
  private isPresenting: boolean = false;

  // XR specific objects
  private cameraRig: THREE.Group;
  private arGroup: THREE.Group; // New group for AR elements
  private controllers: THREE.Group[];
  private controllerGrips: THREE.Group[];
  private controllerModelFactory: XRControllerModelFactory;

  // AR specific objects
  private gridHelper: THREE.GridHelper;
  private groundPlane: THREE.Mesh;
  private hitTestMarker: THREE.Mesh;
  private arLight: THREE.DirectionalLight;
  private hitTestSource: XRHitTestSource | null = null;
  private hitTestSourceRequested = false;

  // Event handlers
  private xrSessionStartCallback: (() => void) | null = null;
  private xrSessionEndCallback: (() => void) | null = null;
  private xrAnimationFrameCallback: ((frame: XRFrame) => void) | null = null;
  private controllerAddedCallback: ((controller: THREE.Group) => void) | null = null;
  private controllerRemovedCallback: ((controller: THREE.Group) => void) | null = null;

  private constructor(sceneManager: SceneManager) {
    this.sceneManager = sceneManager;
    
    // Initialize XR objects
    this.cameraRig = new THREE.Group();
    this.arGroup = new THREE.Group(); // Initialize AR group
    this.controllers = [new THREE.Group(), new THREE.Group()];
    this.controllerGrips = [new THREE.Group(), new THREE.Group()];
    this.controllerModelFactory = new XRControllerModelFactory();

    // Initialize AR objects
    this.gridHelper = this.createGridHelper();
    this.groundPlane = this.createGroundPlane();
    this.hitTestMarker = this.createHitTestMarker();
    this.arLight = this.createARLight();

    this.setupXRObjects();
  }

  static getInstance(sceneManager: SceneManager): XRSessionManager {
    if (!XRSessionManager.instance) {
      XRSessionManager.instance = new XRSessionManager(sceneManager);
    }
    return XRSessionManager.instance;
  }

  private createGridHelper(): THREE.GridHelper {
    const grid = new THREE.GridHelper(10, 10, 0x808080, 0x808080);
    grid.material.transparent = true;
    grid.material.opacity = 0.5;
    grid.position.y = -0.01; // Slightly below ground to avoid z-fighting
    grid.visible = false; // Start hidden until AR session begins
    grid.layers.set(1); // Set to AR layer
    return grid;
  }

  private createGroundPlane(): THREE.Mesh {
    const geometry = new THREE.PlaneGeometry(10, 10);
    const material = new THREE.MeshPhongMaterial({
      color: 0x999999,
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide
    });
    const plane = new THREE.Mesh(geometry, material);
    plane.rotateX(-Math.PI / 2);
    plane.position.y = -0.02; // Below grid
    plane.visible = false; // Start hidden until AR session begins
    plane.layers.set(1); // Set to AR layer
    return plane;
  }

  private createHitTestMarker(): THREE.Mesh {
    const geometry = new THREE.RingGeometry(0.15, 0.2, 32);
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide
    });
    const marker = new THREE.Mesh(geometry, material);
    marker.rotateX(-Math.PI / 2);
    marker.visible = false;
    marker.layers.set(1); // Set to AR layer
    return marker;
  }

  private createARLight(): THREE.DirectionalLight {
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1);
    light.layers.set(1); // Set to AR layer
    return light;
  }

  private setupXRObjects(): void {
    const scene = this.sceneManager.getScene();
    
    // Add camera rig to scene
    scene.add(this.cameraRig);

    // Add AR group to camera rig
    this.cameraRig.add(this.arGroup);

    // Add AR objects to AR group
    this.arGroup.add(this.gridHelper);
    this.arGroup.add(this.groundPlane);
    this.arGroup.add(this.hitTestMarker);
    this.arGroup.add(this.arLight);

    // Setup controllers
    this.controllers.forEach((_controller, index) => {
      this.setupController(index);
    });

    // Setup controller grips
    this.controllerGrips.forEach(grip => {
      this.setupControllerGrip(grip);
    });
  }

  private setupController(index: number): void {
    const controller = this.controllers[index];
    const controllerGrip = this.controllerGrips[index];

    controller.addEventListener('connected', (event: any) => {
      const controllerModel = this.buildController(event.data);
      controller.add(controllerModel);
      this.notifyControllerAdded(controller);
    });

    controller.addEventListener('disconnected', () => {
      controller.remove(...controller.children);
      this.notifyControllerRemoved(controller);
    });

    this.cameraRig.add(controller);
    this.cameraRig.add(controllerGrip);
  }

  private setupControllerGrip(grip: THREE.Group): void {
    const controllerModel = this.controllerModelFactory.createControllerModel(grip);
    grip.add(controllerModel);
  }

  private buildController(_inputSource: XRInputSource): THREE.Group {
    const controller = new THREE.Group();
    const geometry = new THREE.SphereGeometry(0.1, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const sphere = new THREE.Mesh(geometry, material);
    controller.add(sphere);
    return controller;
  }

  /**
   * Initialize XR session
   */
  async initXRSession(): Promise<void> {
    if (this.isPresenting) {
      _logger.warn('XR session already active');
      return;
    }

    if (!platformManager.getCapabilities().xrSupported || !navigator.xr) {
      throw new Error('XR not supported on this platform');
    }

    try {
      const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor', 'hit-test'],
        optionalFeatures: ['hand-tracking', 'layers', 'light-estimation']
      });

      if (!session) {
        throw new Error('Failed to create XR session');
      }

      this.session = session;

      // Setup XR rendering
      const renderer = this.sceneManager.getRenderer();
      await renderer.xr.setSession(this.session);
      
      // Configure renderer for AR
      renderer.xr.enabled = true;
      
      // Clear background for AR passthrough
      const scene = this.sceneManager.getScene();
      scene.background = null;
      
      // Get reference space
      this.referenceSpace = await this.session.requestReferenceSpace('local-floor');
      
      // Setup session event handlers
      this.session.addEventListener('end', this.onXRSessionEnd);

      // Enable AR layer for camera
      const camera = this.sceneManager.getCamera();
      camera.layers.enable(1);

      // Reset camera rig position
      this.cameraRig.position.set(0, 0, 0);
      this.cameraRig.quaternion.identity();

      // Show AR visualization elements after a short delay to ensure proper placement
      setTimeout(() => {
        this.gridHelper.visible = true;
        this.groundPlane.visible = true;
        this.arLight.visible = true;
      }, 1000);
      
      this.isPresenting = true;
      _logger.log('XR session initialized');

      // Notify session start
      if (this.xrSessionStartCallback) {
        this.xrSessionStartCallback();
      }
    } catch (error) {
      _logger.error('Failed to initialize XR session:', error);
      throw error;
    }
  }

  /**
   * End XR session
   */
  async endXRSession(): Promise<void> {
    if (this.session) {
      await this.session.end();
    }
  }

  private onXRSessionEnd = (): void => {
    if (this.hitTestSource) {
      this.hitTestSource.cancel();
      this.hitTestSource = null;
    }
    
    this.session = null;
    this.referenceSpace = null;
    this.hitTestSourceRequested = false;
    this.isPresenting = false;

    // Hide AR visualization elements
    this.gridHelper.visible = false;
    this.groundPlane.visible = false;
    this.hitTestMarker.visible = false;
    this.arLight.visible = false;

    // Reset camera rig
    this.cameraRig.position.set(0, 0, 0);
    this.cameraRig.quaternion.identity();

    // Reset scene background
    const scene = this.sceneManager.getScene();
    scene.background = new THREE.Color(BACKGROUND_COLOR);

    // Disable AR layer for camera
    const camera = this.sceneManager.getCamera();
    camera.layers.disable(1);

    // Reset renderer settings
    const renderer = this.sceneManager.getRenderer();
    renderer.xr.enabled = false;

    _logger.log('XR session ended');

    // Notify session end
    if (this.xrSessionEndCallback) {
      this.xrSessionEndCallback();
    }
  }

  /**
   * XR animation frame
   */
  onXRFrame(frame: XRFrame): void {
    if (!this.session || !this.referenceSpace) return;

    // Get pose
    const pose = frame.getViewerPose(this.referenceSpace);
    if (!pose) return;

    // Let Three.js handle camera updates through WebXRManager
    // Handle hit testing
    this.handleHitTest(frame);

    // Update controller poses
    this.controllers.forEach((controller) => {
      const inputSource = controller.userData.inputSource as XRInputSource;
      if (inputSource) {
        const targetRayPose = frame.getPose(inputSource.targetRaySpace, this.referenceSpace!);
        if (targetRayPose) {
          controller.matrix.fromArray(targetRayPose.transform.matrix);
          controller.matrix.decompose(controller.position, controller.quaternion, controller.scale);
        }
      }
    });

    // Update lighting if available
    if (hasLightEstimate(frame)) {
      const lightEstimate = frame.getLightEstimate();
      if (lightEstimate) {
        this.updateARLighting(lightEstimate);
      }
    }

    // Call animation frame callback
    if (this.xrAnimationFrameCallback) {
      this.xrAnimationFrameCallback(frame);
    }
  }

  private async handleHitTest(frame: XRFrame): Promise<void> {
    if (!this.hitTestSourceRequested && this.session && hasHitTest(this.session)) {
      try {
        const viewerSpace = await this.session.requestReferenceSpace('viewer');
        if (!viewerSpace) {
          throw new Error('Failed to get viewer reference space');
        }

        const hitTestSource = await this.session.requestHitTestSource({
          space: viewerSpace
        });

        if (hitTestSource) {
          this.hitTestSource = hitTestSource;
          this.hitTestSourceRequested = true;
        }
      } catch (error) {
        _logger.error('Failed to initialize hit test source:', error);
        this.hitTestSourceRequested = true; // Prevent further attempts
      }
    }

    if (this.hitTestSource && this.referenceSpace) {
      const hitTestResults = frame.getHitTestResults(this.hitTestSource);
      if (hitTestResults.length > 0) {
        const hit = hitTestResults[0];
        const pose = hit.getPose(this.referenceSpace);
        if (pose) {
          this.hitTestMarker.visible = true;
          this.hitTestMarker.position.set(
            pose.transform.position.x,
            pose.transform.position.y,
            pose.transform.position.z
          );

          // Update grid and ground plane position to match hit test
          this.gridHelper.position.y = pose.transform.position.y;
          this.groundPlane.position.y = pose.transform.position.y - 0.01;
        }
      } else {
        this.hitTestMarker.visible = false;
      }
    }
  }

  private updateARLighting(lightEstimate: XRLightEstimate): void {
    const intensity = lightEstimate.primaryLightIntensity?.value || 1;
    const direction = lightEstimate.primaryLightDirection;
    
    if (direction) {
      this.arLight.position.set(direction.x, direction.y, direction.z);
    }
    this.arLight.intensity = intensity;
  }

  /**
   * Set session event callbacks
   */
  setSessionCallbacks(
    onStart: () => void,
    onEnd: () => void,
    onFrame: (frame: XRFrame) => void
  ): void {
    this.xrSessionStartCallback = onStart;
    this.xrSessionEndCallback = onEnd;
    this.xrAnimationFrameCallback = onFrame;
  }

  public onControllerAdded(callback: (controller: THREE.Group) => void): void {
    this.controllerAddedCallback = callback;
  }

  public onControllerRemoved(callback: (controller: THREE.Group) => void): void {
    this.controllerRemovedCallback = callback;
  }

  private notifyControllerAdded(controller: THREE.Group): void {
    this.controllerAddedCallback?.(controller);
  }

  private notifyControllerRemoved(controller: THREE.Group): void {
    this.controllerRemovedCallback?.(controller);
  }

  /**
   * Get XR objects
   */
  getCameraRig(): THREE.Group {
    return this.cameraRig;
  }

  getControllers(): THREE.Group[] {
    return this.controllers;
  }

  getControllerGrips(): THREE.Group[] {
    return this.controllerGrips;
  }

  /**
   * Check if currently in XR session
   */
  isXRPresenting(): boolean {
    return this.isPresenting;
  }

  /**
   * Get current XR session
   */
  getSession(): XRSession | null {
    return this.session;
  }

  /**
   * Get reference space
   */
  getReferenceSpace(): XRReferenceSpace | null {
    return this.referenceSpace;
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    if (this.session) {
      this.session.end().catch(console.error);
    }

    this.controllers.forEach(controller => {
      controller.removeEventListener('connected', (event: any) => {
        const controllerModel = this.buildController(event.data);
        controller.add(controllerModel);
        this.notifyControllerAdded(controller);
      });

      controller.removeEventListener('disconnected', () => {
        controller.remove(...controller.children);
        this.notifyControllerRemoved(controller);
      });
    });

    this.controllerGrips.forEach(grip => {
      grip.remove(...grip.children);
    });

    this.hitTestSource?.cancel();
    this.hitTestSource = null;
    this.hitTestSourceRequested = false;

    this.session = null;
    this.referenceSpace = null;
    this.isPresenting = false;
  }
}
