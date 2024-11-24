import * as THREE from 'three';
import { XRButton } from 'three/examples/jsm/webxr/XRButton.js';
import { initXRInteraction, handleXRInput } from './xrInteraction.js';

// Constants
const MOVEMENT_SPEED = 0.05;
const XR_SPRITE_SCALE = 0.5;

/**
 * Enhanced XR Session Manager using Three.js WebXR
 */
class XRSessionManager {
    constructor(renderer, scene, camera, effectsManager) {
        this.renderer = renderer;
        this.scene = scene;
        this.camera = camera;
        this.effectsManager = effectsManager;
        this.referenceSpace = null;
        this.originalScales = new WeakMap();
        this.xrInteraction = null;
        this.sessionActive = false;
        this.cameraRig = null;
    }

    /**
     * Initialize XR session manager
     */
    async init() {
        try {
            // Check if XR is supported
            if (!this.renderer.xr) {
                console.warn('WebXR not supported by renderer');
                return;
            }

            // Enable XR on renderer
            this.renderer.xr.enabled = true;

            // Initialize camera rig
            this.initCameraRig();

            // Initialize XR interaction
            this.xrInteraction = initXRInteraction(this.scene, this.camera, this.renderer);

            // Set up session event handlers
            this.setupEventHandlers();

        } catch (error) {
            console.error('Error initializing XR session manager:', error);
        }
    }

    /**
     * Initialize camera rig with proper hierarchy
     */
    initCameraRig() {
        // Create camera rig if it doesn't exist
        if (!this.cameraRig) {
            this.cameraRig = new THREE.Group();
            this.cameraRig.name = 'cameraRig';
        }

        // Create camera offset for height adjustment if not already a child of the rig
        let cameraOffset = this.cameraRig.children.find(child => child.name === 'cameraOffset');
        if (!cameraOffset) {
            cameraOffset = new THREE.Group();
            cameraOffset.name = 'cameraOffset';
            cameraOffset.position.y = 1.6; // Average eye height
            this.cameraRig.add(cameraOffset);
        }

        // Add camera to offset if not already there
        if (!cameraOffset.children.includes(this.camera)) {
            this.camera.name = 'xrCamera';
            cameraOffset.add(this.camera);
        }

        // Add rig to scene if not already there
        if (!this.scene.children.includes(this.cameraRig)) {
            this.scene.add(this.cameraRig);
        }

        // Set initial positions
        this.camera.position.set(0, 0, 0);
        this.cameraRig.position.set(0, 0, 0);
    }

    /**
     * Set up session event handlers
     */
    setupEventHandlers() {
        // Session start handler
        this.renderer.xr.addEventListener('sessionstart', async (event) => {
            console.log('XR session started');
            this.sessionActive = true;

            const session = this.renderer.xr.getSession();
            await this.setupReferenceSpace(session);
            this.handleXRSprites(true);

            // Initialize camera position
            this.cameraRig.position.set(0, 0, 0);
            this.camera.position.set(0, 0, 0);

            window.dispatchEvent(new CustomEvent('xrsessionstart'));
        });

        // Session end handler
        this.renderer.xr.addEventListener('sessionend', () => {
            console.log('XR session ended');
            this.sessionActive = false;
            this.handleXRSprites(false);
            this.resetCameraRig();
            window.dispatchEvent(new CustomEvent('xrsessionend'));
        });
    }

    /**
     * Add XR button to the scene
     * @returns {Promise<void>}
     */
    async addXRButton() {
        try {
            if (!this.renderer.xr.enabled) {
                console.warn('XR not enabled on renderer');
                return;
            }

            const sessionInit = {
                optionalFeatures: [
                    'local-floor',
                    'bounded-floor',
                    'hand-tracking',
                    'layers'
                ]
            };

            // Check if VR is supported
            const isVRSupported = await navigator.xr?.isSessionSupported('immersive-vr');
            
            if (isVRSupported) {
                const button = XRButton.createButton(this.renderer, {
                    mode: 'immersive-vr',
                    sessionInit,
                    onSessionStarted: (session) => this.onSessionStarted(session),
                    onSessionEnded: () => this.onSessionEnded()
                });
                document.body.appendChild(button);
            } else {
                console.warn('VR not supported on this device');
            }
        } catch (error) {
            console.error('Error adding XR button:', error);
        }
    }

    /**
     * Handle session start
     * @param {XRSession} session - The XR session
     */
    async onSessionStarted(session) {
        try {
            await this.setupReferenceSpace(session);
            this.handleXRSprites(true);
        } catch (error) {
            console.error('Error starting XR session:', error);
        }
    }

    /**
     * Handle session end
     */
    onSessionEnded() {
        this.handleXRSprites(false);
        this.resetCameraRig();
    }

    /**
     * Set up reference space with fallback options
     * @param {XRSession} session - The XR session
     */
    async setupReferenceSpace(session) {
        try {
            this.referenceSpace = await session.requestReferenceSpace('local-floor');
            console.log('Using local-floor reference space');
        } catch (error) {
            console.warn('Failed to get local-floor reference space:', error);
            try {
                this.referenceSpace = await session.requestReferenceSpace('local');
                console.log('Falling back to local reference space');
            } catch (error) {
                console.error('Failed to get any reference space:', error);
            }
        }
    }

    /**
     * Handle sprite scaling for XR
     * @param {boolean} enteringXR - Whether entering or exiting XR
     */
    handleXRSprites(enteringXR) {
        this.scene.traverse((object) => {
            if (object.isSprite) {
                if (enteringXR) {
                    this.originalScales.set(object, object.scale.clone());
                    object.scale.multiplyScalar(XR_SPRITE_SCALE);
                    object.layers.enableAll();
                    
                    if (object.material.map) {
                        object.material.map.generateMipmaps = false;
                        object.material.map.minFilter = THREE.LinearFilter;
                        object.material.map.needsUpdate = true;
                    }
                } else {
                    const originalScale = this.originalScales.get(object);
                    if (originalScale) {
                        object.scale.copy(originalScale);
                    }
                    
                    if (object.material.map) {
                        object.material.map.generateMipmaps = true;
                        object.material.map.minFilter = THREE.LinearMipmapLinearFilter;
                        object.material.map.needsUpdate = true;
                    }
                }
            }
        });
    }

    /**
     * Reset camera rig to initial position
     */
    resetCameraRig() {
        if (this.cameraRig) {
            this.cameraRig.position.set(0, 0, 0);
            this.cameraRig.rotation.set(0, 0, 0);
        }
        if (this.camera) {
            this.camera.position.set(0, 0, 0);
            this.camera.rotation.set(0, 0, 0);
        }
    }

    /**
     * Update XR frame
     * @param {number} timestamp - Frame timestamp
     * @param {XRFrame} frame - XR frame
     */
    update(timestamp, frame) {
        if (!this.sessionActive || !frame) return;

        try {
            // Update XR camera pose
            if (this.referenceSpace) {
                const pose = frame.getViewerPose(this.referenceSpace);
                if (pose) {
                    // Update camera rig based on pose
                    const position = pose.transform.position;
                    const orientation = pose.transform.orientation;
                    
                    this.cameraRig.position.set(position.x, position.y, position.z);
                    this.cameraRig.quaternion.set(
                        orientation.x,
                        orientation.y,
                        orientation.z,
                        orientation.w
                    );
                }
            }

            // Update XR interaction
            if (this.xrInteraction) {
                this.xrInteraction.update();
                handleXRInput(frame, this.referenceSpace);
            }

            // Handle input sources
            const session = frame.session;
            for (const inputSource of session.inputSources) {
                if (inputSource.gamepad) {
                    this.handleControllerInput(inputSource.gamepad);
                }
            }

        } catch (error) {
            console.error('Error updating XR frame:', error);
        }
    }

    /**
     * Handle controller input
     * @param {Gamepad} gamepad - The XR gamepad
     */
    handleControllerInput(gamepad) {
        if (!gamepad?.axes || gamepad.axes.length < 2) return;

        try {
            const [x, y] = gamepad.axes;
            const deadzone = 0.1;

            if (Math.abs(x) > deadzone || Math.abs(y) > deadzone) {
                // Get movement direction in camera space
                const forward = new THREE.Vector3();
                this.camera.getWorldDirection(forward);
                forward.y = 0;
                forward.normalize();

                const right = new THREE.Vector3();
                right.crossVectors(new THREE.Vector3(0, 1, 0), forward);

                // Calculate movement
                const movement = new THREE.Vector3();
                movement.addScaledVector(right, x * MOVEMENT_SPEED);
                movement.addScaledVector(forward, -y * MOVEMENT_SPEED);

                // Apply movement to camera rig
                this.cameraRig.position.add(movement);
            }
        } catch (error) {
            console.error('Error handling controller input:', error);
        }
    }

    /**
     * Clean up resources
     */
    dispose() {
        this.originalScales.clear();
        if (this.xrInteraction) {
            this.xrInteraction.cleanup();
        }
    }
}

// Export functions
export function initXRSession(renderer, scene, camera, effectsManager) {
    // Check if renderer has XR capability
    if (!renderer.xr) {
        console.warn('WebXR not supported by renderer');
        return null;
    }

    const xrSessionManager = new XRSessionManager(renderer, scene, camera, effectsManager);
    xrSessionManager.init();
    return xrSessionManager;
}

/**
 * Add XR button to enable VR mode
 * @param {XRSessionManager} xrSessionManager - The XR session manager
 * @returns {Promise<void>}
 */
export async function addXRButton(xrSessionManager) {
    if (!xrSessionManager) {
        console.warn('XR session manager not initialized');
        return;
    }
    await xrSessionManager.addXRButton();
}

export function handleXRSession(renderer, scene, camera, xrSessionManager, effectsManager) {
    if (!xrSessionManager) return;

    renderer.setAnimationLoop((timestamp, frame) => {
        // Update XR session
        xrSessionManager.update(timestamp, frame);

        // Render scene
        if (effectsManager) {
            effectsManager.animate();
            effectsManager.render();
        } else {
            renderer.render(scene, camera);
        }
    });
}

export function updateXRFrame(renderer, scene, camera, xrSessionManager, effectsManager) {
    if (!xrSessionManager?.sessionActive) return;

    // Update XR session
    const frame = renderer.xr.getFrame();
    xrSessionManager.update(performance.now(), frame);

    // Render scene
    if (effectsManager) {
        effectsManager.animate();
        effectsManager.render();
    } else {
        renderer.render(scene, camera);
    }
}
