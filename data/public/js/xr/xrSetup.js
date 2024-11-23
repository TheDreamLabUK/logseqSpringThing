import * as THREE from 'three';
import { XRButton } from 'three/examples/jsm/webxr/XRButton.js';
import { initXRInteraction, handleXRInput } from './xrInteraction.js';

// Constants
const MOVEMENT_SPEED = 0.05;
const XR_SPRITE_SCALE = 0.5;

/**
 * Initializes the WebXR session for immersive experiences.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 */
export function initXRSession(renderer, scene, camera) {
    if (!scene || !camera) {
        console.error('Scene or camera not provided to initXRSession');
        return;
    }

    // Store original sprite scales for restoration
    const originalScales = new WeakMap();

    // Initialize hand tracking with enhanced features
    const xrInteraction = initXRInteraction(scene, camera, renderer);

    // Configure renderer for XR with optimized settings
    renderer.xr.enabled = true;
    renderer.xr.setFramebufferScaleFactor(1.0); // Optimize resolution
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    renderer.alpha = true;

    /**
     * Handles sprite scaling and visibility for XR
     * @param {boolean} enteringXR - Whether entering or exiting XR
     */
    function handleXRSprites(enteringXR) {
        scene.traverse((object) => {
            if (object.isSprite) {
                if (enteringXR) {
                    // Store original scale
                    originalScales.set(object, object.scale.clone());
                    
                    // Scale for XR
                    object.scale.multiplyScalar(XR_SPRITE_SCALE);
                    object.layers.enableAll();
                    
                    // Optimize sprite texture
                    if (object.material.map) {
                        object.material.map.generateMipmaps = false;
                        object.material.map.minFilter = THREE.LinearFilter;
                        object.material.map.needsUpdate = true;
                    }
                } else {
                    // Restore original scale
                    const originalScale = originalScales.get(object);
                    if (originalScale) {
                        object.scale.copy(originalScale);
                    }
                    
                    // Reset texture settings
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
     * Creates XR session configuration
     * @param {string} mode - XR session mode
     * @returns {Object} Session configuration
     */
    function createSessionConfig(mode) {
        return {
            mode: mode,
            sessionInit: {
                optionalFeatures: [
                    'dom-overlay',
                    'local-floor',
                    'bounded-floor',
                    'hand-tracking',
                    'layers',
                    mode === 'immersive-ar' ? 'passthrough' : null
                ].filter(Boolean),
                domOverlay: { root: document.body }
            },
            onSessionStarted: (session) => {
                console.log(`${mode} session started`);
                handleXRSprites(true);
                
                session.addEventListener('end', () => {
                    console.log(`${mode} session ended`);
                    handleXRSprites(false);
                    window.dispatchEvent(new CustomEvent('xrsessionend'));
                });

                // Request reference space with fallback
                requestReferenceSpace(session, renderer);
                
                window.dispatchEvent(new CustomEvent('xrsessionstart'));
            },
            onSessionEnded: () => {
                console.log(`${mode} session cleanup`);
                handleXRSprites(false);
                
                // Clear any cached resources
                originalScales.clear();
                
                // Force renderer reset
                renderer.setPixelRatio(window.devicePixelRatio);
                renderer.setSize(window.innerWidth, window.innerHeight);
            }
        };
    }

    if ('xr' in navigator) {
        // Check for AR support first
        navigator.xr.isSessionSupported('immersive-ar')
            .then(arSupported => {
                if (arSupported) {
                    const xrButton = XRButton.createButton(renderer, createSessionConfig('immersive-ar'));
                    document.body.appendChild(xrButton);
                } else {
                    // Fall back to VR if AR is not supported
                    return navigator.xr.isSessionSupported('immersive-vr')
                        .then(vrSupported => {
                            if (vrSupported) {
                                const xrButton = XRButton.createButton(renderer, createSessionConfig('immersive-vr'));
                                document.body.appendChild(xrButton);
                            } else {
                                console.warn('Neither AR nor VR is supported');
                            }
                        });
                }
            })
            .catch(err => {
                console.error('Error checking XR session support:', err);
            });

        // Add session event listeners
        renderer.xr.addEventListener('sessionstart', (event) => {
            console.log('XR session started');
            const session = event.target.getSession();
            requestReferenceSpace(session, renderer);
        });

        renderer.xr.addEventListener('sessionend', () => {
            console.log('XR session ended');
            // Force a renderer reset
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    } else {
        console.warn('WebXR not supported in this browser.');
    }

    // Handle window resizes
    window.addEventListener('resize', () => {
        if (!renderer.xr.isPresenting) {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
    });

    return xrInteraction;
}

/**
 * Request reference space with fallback options
 * @param {XRSession} session - The XR session
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer
 */
async function requestReferenceSpace(session, renderer) {
    try {
        const refSpace = await session.requestReferenceSpace('local-floor');
        console.log('Got local-floor reference space');
        renderer.xr.setReferenceSpace(refSpace);
    } catch (err) {
        console.warn('Failed to get local-floor reference space:', err);
        try {
            const refSpace = await session.requestReferenceSpace('local');
            console.log('Falling back to local reference space');
            renderer.xr.setReferenceSpace(refSpace);
        } catch (err) {
            console.error('Failed to get any reference space:', err);
        }
    }
}

/**
 * Updates camera position based on XR pose with error handling
 * @param {XRFrame} frame - The XR frame
 * @param {XRReferenceSpace} refSpace - The XR reference space
 * @param {THREE.Camera} camera - The Three.js camera
 */
function updateCameraFromXRPose(frame, refSpace, camera) {
    if (!frame || !refSpace || !camera) return;

    try {
        const pose = frame.getViewerPose(refSpace);
        if (pose) {
            const view = pose.views[0];
            if (view) {
                const position = new THREE.Vector3(
                    view.transform.position.x,
                    view.transform.position.y,
                    view.transform.position.z
                );
                camera.position.copy(position);
            }
        }
    } catch (error) {
        console.error('Error updating camera from XR pose:', error);
    }
}

/**
 * Handles the XR session's rendering loop with error recovery
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer
 * @param {THREE.Scene} scene - The Three.js scene
 * @param {THREE.Camera} camera - The Three.js camera
 * @param {Object} xrInteraction - The XR interaction instance
 */
export function handleXRSession(renderer, scene, camera, xrInteraction) {
    if (!renderer || !scene || !camera) {
        console.error('Required parameters missing in handleXRSession');
        return;
    }

    let frameCount = 0;
    const MAX_ERRORS = 5;
    let errorCount = 0;

    renderer.setAnimationLoop((timestamp, frame) => {
        if (frame) {
            try {
                frameCount++;
                const session = renderer.xr.getSession();
                const refSpace = renderer.xr.getReferenceSpace();

                if (session && refSpace) {
                    updateCameraFromXRPose(frame, refSpace, camera);

                    // Update hand tracking and interactions
                    if (xrInteraction) {
                        xrInteraction.update();
                        handleXRInput(frame, refSpace);
                    }

                    // Handle input sources
                    for (const source of session.inputSources) {
                        if (source?.gamepad?.handedness === 'left') {
                            handleGamepadInput(source.gamepad, camera);
                        }
                    }

                    // Reset error count on successful frames
                    if (frameCount % 60 === 0) {
                        errorCount = 0;
                    }
                }
            } catch (error) {
                console.error('Error in XR frame:', error);
                errorCount++;
                
                // End session if too many errors occur
                if (errorCount >= MAX_ERRORS) {
                    console.error('Too many XR errors, ending session');
                    renderer.xr.getSession()?.end();
                    return;
                }
            }
        }
        
        // Render the scene
        try {
            renderer.render(scene, camera);
        } catch (error) {
            console.error('Error rendering scene:', error);
        }
    });
}

/**
 * Handles gamepad input in XR with improved movement
 * @param {Gamepad} gamepad - The XR gamepad
 * @param {THREE.Camera} camera - The Three.js camera
 */
function handleGamepadInput(gamepad, camera) {
    if (!gamepad || !camera || !camera.parent) return;

    try {
        if (gamepad.axes.length >= 2) {
            const [x, y] = gamepad.axes;
            const deadzone = 0.1;

            if (Math.abs(x) > deadzone || Math.abs(y) > deadzone) {
                const forward = new THREE.Vector3();
                camera.getWorldDirection(forward);
                forward.y = 0;
                forward.normalize();

                const right = new THREE.Vector3();
                right.crossVectors(new THREE.Vector3(0, 1, 0), forward);

                const movement = new THREE.Vector3();
                movement.addScaledVector(right, x * MOVEMENT_SPEED);
                movement.addScaledVector(forward, -y * MOVEMENT_SPEED);

                const userGroup = camera.parent;
                userGroup.position.add(movement);
            }
        }
    } catch (error) {
        console.error('Error handling gamepad input:', error);
    }
}

/**
 * Updates the XR frame with error handling
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer
 * @param {THREE.Scene} scene - The Three.js scene
 * @param {THREE.Camera} camera - The Three.js camera
 * @param {Object} xrInteraction - The XR interaction instance
 */
export function updateXRFrame(renderer, scene, camera, xrInteraction) {
    if (renderer.xr.isPresenting) {
        try {
            const session = renderer.xr.getSession();
            if (session && xrInteraction) {
                xrInteraction.update();
            }
        } catch (error) {
            console.error('Error updating XR frame:', error);
        }
    }
    
    try {
        renderer.render(scene, camera);
    } catch (error) {
        console.error('Error rendering scene:', error);
    }
}
