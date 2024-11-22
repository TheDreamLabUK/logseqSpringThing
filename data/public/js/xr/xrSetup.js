// public/js/xr/xrSetup.js

import * as THREE from 'three';
import { XRButton } from 'three/examples/jsm/webxr/XRButton.js';

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

    // Configure renderer for XR
    renderer.xr.enabled = true;
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0);
    renderer.alpha = true;

    if ('xr' in navigator) {
        navigator.xr.isSessionSupported('immersive-vr')
            .then(vrSupported => {
                if (vrSupported) {
                    const sessionInit = {
                        optionalFeatures: ['local-floor', 'bounded-floor']
                    };

                    const xrButton = XRButton.createButton(renderer, {
                        mode: 'immersive-vr',
                        sessionInit: sessionInit,
                        onSessionStarted: (session) => {
                            console.log('VR session started');
                            session.addEventListener('end', () => {
                                console.log('VR session ended');
                                window.dispatchEvent(new CustomEvent('xrsessionend'));
                            });
                            window.dispatchEvent(new CustomEvent('xrsessionstart'));
                        },
                        onSessionEnded: () => {
                            console.log('VR session cleanup');
                        }
                    });

                    document.body.appendChild(xrButton);
                } else {
                    return navigator.xr.isSessionSupported('immersive-ar')
                        .then(arSupported => {
                            if (arSupported) {
                                const sessionInit = {
                                    optionalFeatures: ['dom-overlay'],
                                    domOverlay: { root: document.body }
                                };

                                const xrButton = XRButton.createButton(renderer, {
                                    mode: 'immersive-ar',
                                    sessionInit: sessionInit
                                });

                                document.body.appendChild(xrButton);
                            } else {
                                console.warn('Neither VR nor AR is supported');
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
        // Get the position from the first view (center eye in VR)
        const view = pose.views[0];
        if (view) {
            const position = new THREE.Vector3();
            position.set(
                view.transform.position.x,
                view.transform.position.y,
                view.transform.position.z
            );
            
            // Update the camera's local position within its parent group
            camera.position.copy(position);
            
            console.log('XR Pose Update - Camera Position:', 
                position.toArray().map(v => v.toFixed(3)),
                'Parent Position:', 
                camera.parent ? camera.parent.position.toArray().map(v => v.toFixed(3)) : 'No parent');
        }
    }
}

/**
 * Handles the XR session's rendering loop.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 */
export function handleXRSession(renderer, scene, camera) {
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
                    // Update camera position from XR pose
                    updateCameraFromXRPose(frame, refSpace, camera);

                    // Handle input sources
                    for (const source of session.inputSources) {
                        if (source && source.gamepad && source.handedness === 'left') {
                            handleGamepadInput(source.gamepad, camera);
                            console.log('Processing left controller input:', 
                                source.gamepad.axes[0].toFixed(2), 
                                source.gamepad.axes[1].toFixed(2));
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

    // Handle joystick movement
    if (gamepad.axes.length >= 2) {
        const [x, y] = gamepad.axes;

        // Only process if joystick is moved significantly
        if (Math.abs(x) > 0.1 || Math.abs(y) > 0.1) {
            console.log('Joystick input:', x.toFixed(2), y.toFixed(2));

            // Get camera's forward and right vectors
            const forward = new THREE.Vector3();
            camera.getWorldDirection(forward);
            forward.y = 0; // Keep movement horizontal
            forward.normalize();

            const right = new THREE.Vector3();
            right.crossVectors(new THREE.Vector3(0, 1, 0), forward);

            // Calculate movement
            const movement = new THREE.Vector3();
            movement.addScaledVector(right, x * MOVEMENT_SPEED);
            movement.addScaledVector(forward, -y * MOVEMENT_SPEED);

            // Move the camera's parent (user group)
            const userGroup = camera.parent;
            const oldPosition = userGroup.position.clone();
            userGroup.position.add(movement);
            
            console.log('Movement:', 
                movement.toArray().map(v => v.toFixed(2)),
                'Old pos:', oldPosition.toArray().map(v => v.toFixed(2)),
                'New pos:', userGroup.position.toArray().map(v => v.toFixed(2)));
        }
    }
}

/**
 * Updates the XR frame, if necessary.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 */
export function updateXRFrame(renderer, scene, camera) {
    if (renderer.xr.isPresenting) {
        try {
            const session = renderer.xr.getSession();
            if (session) {
                // Additional frame updates can be handled here
            }
        } catch (error) {
            console.error('Error updating XR frame:', error);
        }
    }
    renderer.render(scene, camera);
}
