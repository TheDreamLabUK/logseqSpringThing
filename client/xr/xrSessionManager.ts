/**
 * XR session management and rendering
 */

import * as THREE from 'three';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory';
import { createLogger } from '../core/utils';
import { platformManager } from '../platform/platformManager';
import { SceneManager } from '../rendering/scene';

const _logger = createLogger('XRSessionManager');

// XR event types
interface XRControllerEvent extends THREE.Event {
  type: 'connected' | 'disconnected';
  data: XRInputSource;
}

declare module 'three' {
  interface Object3DEventMap {
    connected: XRControllerEvent;
    disconnected: XRControllerEvent;
  }
}

export class XRSessionManager {
  private static instance: XRSessionManager;
  private sceneManager: SceneManager;
  private session: XRSession | null = null;
  private referenceSpace: XRReferenceSpace | null = null;
  private isPresenting: boolean = false;

  // XR specific objects
  private cameraRig: THREE.Group;
  private controllers: THREE.Group[];
  private controllerGrips: THREE.Group[];
  private controllerModelFactory: XRControllerModelFactory;

  // Event handlers
  private xrSessionStartCallback: (() => void) | null = null;
  private xrSessionEndCallback: (() => void) | null = null;
  private xrAnimationFrameCallback: ((frame: XRFrame) => void) | null = null;

  private constructor(sceneManager: SceneManager) {
    this.sceneManager = sceneManager;
    
    // Initialize XR objects
    this.cameraRig = new THREE.Group();
    this.controllers = [new THREE.Group(), new THREE.Group()];
    this.controllerGrips = [new THREE.Group(), new THREE.Group()];
    this.controllerModelFactory = new XRControllerModelFactory();

    this.setupXRObjects();
  }

  static getInstance(sceneManager: SceneManager): XRSessionManager {
    if (!XRSessionManager.instance) {
      XRSessionManager.instance = new XRSessionManager(sceneManager);
    }
    return XRSessionManager.instance;
  }

  private setupXRObjects(): void {
    // Add camera rig to scene
    this.sceneManager.getScene().add(this.cameraRig);

    // Setup controllers
    this.controllers.forEach((controller, _index) => {
      this.cameraRig.add(controller);
      this.setupController(controller);
    });

    // Setup controller grips
    this.controllerGrips.forEach((grip, _index) => {
      this.cameraRig.add(grip);
      this.setupControllerGrip(grip);
    });
  }

  private setupController(controller: THREE.Group): void {
    controller.addEventListener('connected', (event: XRControllerEvent) => {
      controller.userData.inputSource = event.data;

      // Add visual representation of controller
      if (event.data.targetRayMode === 'tracked-pointer') {
        controller.add(this.createControllerPointer());
      }
    });

    controller.addEventListener('disconnected', () => {
      controller.remove(...controller.children);
    });
  }

  private setupControllerGrip(grip: THREE.Group): void {
    const controllerModel = this.controllerModelFactory.createControllerModel(grip);
    grip.add(controllerModel);
  }

  private createControllerPointer(): THREE.Mesh {
    const geometry = new THREE.CylinderGeometry(0.01, 0.02, 0.08);
    geometry.rotateX(-Math.PI / 2);
    const material = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.8
    });
    return new THREE.Mesh(geometry, material);
  }

  /**
   * Initialize XR session
   */
  async initXRSession(): Promise<void> {
    if (!platformManager.getCapabilities().xrSupported || !navigator.xr) {
      throw new Error('XR not supported on this platform');
    }

    try {
      const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor', 'hit-test'],
        optionalFeatures: ['hand-tracking', 'layers']
      });

      if (!session) {
        throw new Error('Failed to create XR session');
      }

      this.session = session;

      // Setup XR rendering
      await this.sceneManager.getRenderer().xr.setSession(this.session);
      
      // Get reference space
      this.referenceSpace = await this.session.requestReferenceSpace('local-floor');
      
      // Setup session event handlers
      this.session.addEventListener('end', () => this.onXRSessionEnd());
      
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

  private onXRSessionEnd(): void {
    this.session = null;
    this.referenceSpace = null;
    this.isPresenting = false;
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

    // Call animation frame callback
    if (this.xrAnimationFrameCallback) {
      this.xrAnimationFrameCallback(frame);
    }
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
      this.session.end();
    }

    this.controllers.forEach(controller => {
      controller.remove(...controller.children);
    });

    this.controllerGrips.forEach(grip => {
      grip.remove(...grip.children);
    });

    this.cameraRig.remove(...this.cameraRig.children);
    this.sceneManager.getScene().remove(this.cameraRig);
  }
}
