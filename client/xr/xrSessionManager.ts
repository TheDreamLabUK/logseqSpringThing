/**
 * XR session management and rendering
 */

import {
    Group,
    GridHelper,
    PlaneGeometry,
    MeshPhongMaterial,
    Mesh,
    RingGeometry,
    MeshBasicMaterial,
    DirectionalLight,
    SphereGeometry,
    Color,
    DoubleSide
} from 'three';
import { XRControllerModelFactory } from 'three/examples/jsm/webxr/XRControllerModelFactory';
import { createLogger } from '../core/utils';
import { platformManager } from '../platform/platformManager';
import { SceneManager } from '../rendering/scene';
import { BACKGROUND_COLOR } from '../core/constants';
import { ControlPanel } from '../ui/ControlPanel';

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
    private cameraRig: Group;
    private arGroup: Group; // New group for AR elements
    private controllers: Group[];
    private controllerGrips: Group[];
    private controllerModelFactory: XRControllerModelFactory;

    // AR specific objects
    private gridHelper: GridHelper;
    private groundPlane: Mesh;
    private hitTestMarker: Mesh;
    private arLight: DirectionalLight;
    private hitTestSource: XRHitTestSource | null = null;
    private hitTestSourceRequested = false;

    // Event handlers
    private xrSessionStartCallback: (() => void) | null = null;
    private xrSessionEndCallback: (() => void) | null = null;
    private xrAnimationFrameCallback: ((frame: XRFrame) => void) | null = null;
    private controllerAddedCallback: ((controller: Group) => void) | null = null;
    private controllerRemovedCallback: ((controller: Group) => void) | null = null;

    constructor(sceneManager: SceneManager) {
        this.sceneManager = sceneManager;
        
        // Initialize XR objects
        this.cameraRig = new Group();
        this.arGroup = new Group(); // Initialize AR group
        this.controllers = [new Group(), new Group()];
        this.controllerGrips = [new Group(), new Group()];
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

    private createGridHelper(): GridHelper {
        const grid = new GridHelper(10, 10, 0x808080, 0x808080);
        grid.material.transparent = true;
        grid.material.opacity = 0.5;
        grid.position.y = -0.01; // Slightly below ground to avoid z-fighting
        grid.visible = false; // Start hidden until AR session begins
        grid.layers.set(1); // Set to AR layer
        return grid;
    }

    private createGroundPlane(): Mesh {
        const geometry = new PlaneGeometry(10, 10);
        const material = new MeshPhongMaterial({
            color: 0x999999,
            transparent: true,
            opacity: 0.3,
            side: DoubleSide
        });
        const plane = new Mesh(geometry, material);
        plane.rotateX(-Math.PI / 2);
        plane.position.y = -0.02; // Below grid
        plane.visible = false; // Start hidden until AR session begins
        plane.layers.set(1); // Set to AR layer
        return plane;
    }

    private createHitTestMarker(): Mesh {
        const geometry = new RingGeometry(0.15, 0.2, 32);
        const material = new MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.8,
            side: DoubleSide
        });
        const marker = new Mesh(geometry, material);
        marker.rotateX(-Math.PI / 2);
        marker.visible = false;
        marker.layers.set(1); // Set to AR layer
        return marker;
    }

    private createARLight(): DirectionalLight {
        const light = new DirectionalLight(0xffffff, 1);
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

    private setupControllerGrip(grip: Group): void {
        const controllerModel = this.controllerModelFactory.createControllerModel(grip);
        grip.add(controllerModel);
    }

    private buildController(_inputSource: XRInputSource): Group {
        const controller = new Group();
        const geometry = new SphereGeometry(0.1, 16, 16);
        const material = new MeshBasicMaterial({ color: 0xffffff });
        const sphere = new Mesh(geometry, material);
        controller.add(sphere);
        return controller;
    }

    async initXRSession(): Promise<void> {
        if (this.isPresenting) {
            _logger.warn('XR session already active');
            return;
        }

        if (!platformManager.getCapabilities().xrSupported || !navigator.xr) {
            throw new Error('XR not supported on this platform');
        }

        try {
            // Check if session mode is supported
            const mode = platformManager.isQuest() ? 'immersive-ar' : 'immersive-vr';
            const isSupported = await navigator.xr.isSessionSupported(mode);
            
            if (!isSupported) {
                throw new Error(`${mode} not supported on this device`);
            }
            
            // Configure features based on mode and platform
            const requiredFeatures = ['local-floor'];
            const optionalFeatures = ['hand-tracking', 'layers'];
            
            // Add mode-specific features for Quest
            if (platformManager.isQuest()) {
                // For Quest AR, require hit-test and make plane detection optional
                requiredFeatures.push('hit-test');
                optionalFeatures.push(
                    'light-estimation',
                    'plane-detection',
                    'anchors',
                    'depth-sensing',
                    'dom-overlay'
                );
            }
            
            // Request session with configured features
            const sessionInit: XRSessionInit = {
                requiredFeatures,
                optionalFeatures,
                domOverlay: platformManager.isQuest() ? { root: document.body } : undefined
            };
            
            _logger.info('Requesting XR session with config:', {
                mode,
                features: sessionInit
            });
            
            const session = await navigator.xr.requestSession(mode, sessionInit);

            if (!session) {
                throw new Error('Failed to create XR session');
            }

            this.session = session;

            // Setup XR rendering
            const renderer = this.sceneManager.getRenderer();
            await renderer.xr.setSession(this.session);
            
            // Configure renderer for AR
            renderer.xr.enabled = true;
            
            // Set up scene for XR mode
            const scene = this.sceneManager.getScene();
            if (platformManager.isQuest()) {
                // Clear background for AR passthrough
                scene.background = null;
            } else {
                // Keep background for VR mode
                scene.background = new Color(BACKGROUND_COLOR);
            }
            
            // Get reference space based on platform
            const spaceType = platformManager.isQuest() ? 'local-floor' : 'bounded-floor';
            this.referenceSpace = await this.session.requestReferenceSpace(spaceType);
            
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

            // Hide control panel in XR mode
            const controlPanel = ControlPanel.getInstance();
            if (controlPanel) {
                controlPanel.hide();
            }

            // Notify session start
            if (this.xrSessionStartCallback) {
                this.xrSessionStartCallback();
            }
        } catch (error) {
            _logger.error('Failed to initialize XR session:', error);
            throw error;
        }
    }

    async endXRSession(): Promise<void> {
        if (this.session) {
            await this.session.end();
        }
    }

    private onXRSessionEnd = (): void => {
        // Clean up hit test source
        if (this.hitTestSource) {
            this.hitTestSource.cancel();
            this.hitTestSource = null;
        }
        
        // Reset session state
        this.session = null;
        this.referenceSpace = null;
        this.hitTestSourceRequested = false;
        this.isPresenting = false;

        // Hide AR visualization elements if in Quest mode
        if (platformManager.isQuest()) {
            this.gridHelper.visible = false;
            this.groundPlane.visible = false;
            this.hitTestMarker.visible = false;
            this.arLight.visible = false;
        }

        // Reset camera and scene
        this.cameraRig.position.set(0, 0, 0);
        this.cameraRig.quaternion.identity();

        // Reset scene background
        const scene = this.sceneManager.getScene();
        scene.background = new Color(BACKGROUND_COLOR);

        // Reset camera layers
        const camera = this.sceneManager.getCamera();
        camera.layers.disable(1); // AR layer

        // Reset renderer
        const renderer = this.sceneManager.getRenderer();
        renderer.xr.enabled = false;

        _logger.log('XR session ended');

        // Show control panel again and notify session end
        ControlPanel.getInstance()?.show();
        this.xrSessionEndCallback?.();

        _logger.log('XR session ended');

        // Show control panel again
        const controlPanel = ControlPanel.getInstance();
        if (controlPanel) {
            controlPanel.show();
        }

        // Notify session end
        if (this.xrSessionEndCallback) {
            this.xrSessionEndCallback();
        }
    }

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

    setSessionCallbacks(
        onStart: () => void,
        onEnd: () => void,
        onFrame: (frame: XRFrame) => void
    ): void {
        this.xrSessionStartCallback = onStart;
        this.xrSessionEndCallback = onEnd;
        this.xrAnimationFrameCallback = onFrame;
    }

    onControllerAdded(callback: (controller: Group) => void): void {
        this.controllerAddedCallback = callback;
    }

    onControllerRemoved(callback: (controller: Group) => void): void {
        this.controllerRemovedCallback = callback;
    }

    private notifyControllerAdded(controller: Group): void {
        this.controllerAddedCallback?.(controller);
    }

    private notifyControllerRemoved(controller: Group): void {
        this.controllerRemovedCallback?.(controller);
    }

    getCameraRig(): Group {
        return this.cameraRig;
    }

    getControllers(): Group[] {
        return this.controllers;
    }

    getControllerGrips(): Group[] {
        return this.controllerGrips;
    }

    isXRPresenting(): boolean {
        return this.isPresenting;
    }

    getSession(): XRSession | null {
        return this.session;
    }

    getReferenceSpace(): XRReferenceSpace | null {
        return this.referenceSpace;
    }

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
