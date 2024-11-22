import * as THREE from 'three';
import { XRButton } from 'three/examples/jsm/webxr/XRButton.js';
import { initXRInteraction, handleXRInput } from './xrInteraction.js';

// Movement speed constant
const MOVEMENT_SPEED = 0.05;

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

    // Initialize hand tracking with enhanced features
    const xrInteraction = initXRInteraction(scene, camera, renderer);

    // Configure renderer for XR
    renderer.xr.enabled = true;
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    renderer.alpha = true;

    if ('xr' in navigator) {
        // Check for AR support first
        navigator.xr.isSessionSupported('immersive-ar')
            .then(arSupported => {
                if (arSupported) {
                    const sessionInit = {
                        optionalFeatures: [
                            'dom-overlay',
                            'local-floor',
                            'bounded-floor',
                            'hand-tracking',
                            'layers',
                            'passthrough'  // Enable passthrough for Quest 3
                        ],
                        domOverlay: { root: document.body }
                    };

                    const xrButton = XRButton.createButton(renderer, {
                        mode: 'immersive-ar',
                        sessionInit: sessionInit,
                        onSessionStarted: (session) => {
                            console.log('AR session started');
                            // Ensure all sprites (labels) are visible in XR
                            scene.traverse((object) => {
                                if (object.isSprite) {
                                    object.layers.enableAll();
                                    // Adjust sprite scale for better visibility in XR
                                    const currentScale = object.scale.clone();
                                    object.scale.set(currentScale.x * 0.5, currentScale.y * 0.5, 1);
                                }
                            });
                            session.addEventListener('end', () => {
                                console.log('AR session ended');
                                window.dispatchEvent(new CustomEvent('xrsessionend'));
                            });
                            window.dispatchEvent(new CustomEvent('xrsessionstart'));
                        },
                        onSessionEnded: () => {
                            console.log('AR session cleanup');
                            // Reset sprite scales
                            scene.traverse((object) => {
                                if (object.isSprite) {
                                    const currentScale = object.scale.clone();
                                    object.scale.set(currentScale.x * 2, currentScale.y * 2, 1);
                                }
                            });
                        }
                    });

                    document.body.appendChild(xrButton);
                } else {
                    // Fall back to VR if AR is not supported
                    return navigator.xr.isSessionSupported('immersive-vr')
                        .then(vrSupported => {
                            if (vrSupported) {
                                const sessionInit = {
                                    optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking']
                                };

                                const xrButton = XRButton.createButton(renderer, {
                                    mode: 'immersive-vr',
                                    sessionInit: sessionInit,
                                    onSessionStarted: (session) => {
                                        console.log('VR session started');
                                        // Apply same label visibility settings in VR
                                        scene.traverse((object) => {
                                            if (object.isSprite) {
                                                object.layers.enableAll();
                                                const currentScale = object.scale.clone();
                                                object.scale.set(currentScale.x * 0.5, currentScale.y * 0.5, 1);
                                            }
                                        });
                                        session.addEventListener('end', () => {
                                            console.log('VR session ended');
                                            window.dispatchEvent(new CustomEvent('xrsessionend'));
                                        });
                                        window.dispatchEvent(new CustomEvent('xrsessionstart'));
                                    },
                                    onSessionEnded: () => {
                                        console.log('VR session cleanup');
                                        // Reset sprite scales
                                        scene.traverse((object) => {
                                            if (object.isSprite) {
                                                const currentScale = object.scale.clone();
                                                object.scale.set(currentScale.x * 2, currentScale.y * 2, 1);
                                            }
                                        });
                                    }
                                });

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

        renderer.xr.addEventListener('sessionstart', (event) => {
            console.log('XR session started');
            const session = event.target.getSession();
            
            session.requestReferenceSpace('local-floor').then(refSpace => {
                console.log('Got local-floor reference space');
                renderer.xr.setReferenceSpace(refSpace);
            }).catch(err => {
                console.warn('Failed to get local-floor reference space:', err);
                session.requestReferenceSpace('local').then(refSpace => {
                    console.log('Falling back to local reference space');
                    renderer.xr.setReferenceSpace(refSpace);
                });
            });
        });

        renderer.xr.addEventListener('sessionend', () => {
            console.log('XR session ended');
        });
    } else {
        console.warn('WebXR not supported in this browser.');
    }

    window.addEventListener('resize', () => {
        if (camera) {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
    });

    return xrInteraction;
}

/**
 * Updates camera position based on XR pose
 * @param {XRFrame} frame - The XR frame
 * @param {XRReferenceSpace} refSpace - The XR reference space
 * @param {THREE.Camera} camera - The Three.js camera
 */
function updateCameraFromXRPose(frame, refSpace, camera) {
    if (!frame || !refSpace || !camera) return;

    const pose = frame.getViewerPose(refSpace);
    if (pose) {
        const view = pose.views[0];
        if (view) {
            const position = new THREE.Vector3();
            position.set(
                view.transform.position.x,
                view.transform.position.y,
                view.transform.position.z
            );
            camera.position.copy(position);
        }
    }
}

/**
 * Handles the XR session's rendering loop.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 * @param {Object} xrInteraction - The XR interaction instance.
 */
export function handleXRSession(renderer, scene, camera, xrInteraction) {
    if (!renderer || !scene || !camera) {
        console.error('Required parameters missing in handleXRSession');
        return;
    }

    renderer.setAnimationLoop((timestamp, frame) => {
        if (frame) {
            try {
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
                        if (source && source.gamepad && source.handedness === 'left') {
                            handleGamepadInput(source.gamepad, camera);
                        }
                    }
                }
            } catch (error) {
                console.error('Error in XR frame:', error);
            }
        }
        renderer.render(scene, camera);
    });
}

/**
 * Handles gamepad input in XR.
 * @param {Gamepad} gamepad - The XR gamepad object.
 * @param {THREE.Camera} camera - The Three.js camera.
 */
function handleGamepadInput(gamepad, camera) {
    if (!gamepad || !camera || !camera.parent) return;

    if (gamepad.axes.length >= 2) {
        const [x, y] = gamepad.axes;

        if (Math.abs(x) > 0.1 || Math.abs(y) > 0.1) {
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
}

/**
 * Updates the XR frame, if necessary.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 * @param {Object} xrInteraction - The XR interaction instance.
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
    renderer.render(scene, camera);
}
