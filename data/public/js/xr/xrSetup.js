// public/js/xr/xrSetup.js

import * as THREE from 'three';
import { XRButton } from 'three/addons/webxr/XRButton.js';

/**
 * Initializes the WebXR session for immersive experiences.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 */
export function initXRSession(renderer, scene, camera) {
    // Configure renderer for XR
    renderer.xr.enabled = true;
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0x000000, 0); // Transparent background for AR
    renderer.alpha = true; // Enable alpha for AR passthrough

    if ('xr' in navigator) {
        // Check for VR support first
        navigator.xr.isSessionSupported('immersive-vr')
            .then(vrSupported => {
                if (vrSupported) {
                    // VR is supported, create VR button
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
                    // If VR is not supported, check for AR
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

        // Set up session start event handler
        renderer.xr.addEventListener('sessionstart', (event) => {
            console.log('XR session started');
            const session = event.target.getSession();
            
            // Configure reference space
            session.requestReferenceSpace('local-floor').then(refSpace => {
                renderer.xr.setReferenceSpace(refSpace);
            }).catch(err => {
                console.warn('Failed to get local-floor reference space:', err);
                // Fallback to local reference space
                session.requestReferenceSpace('local').then(refSpace => {
                    renderer.xr.setReferenceSpace(refSpace);
                });
            });
        });

        // Set up session end event handler
        renderer.xr.addEventListener('sessionend', () => {
            console.log('XR session ended');
        });
    } else {
        console.warn('WebXR not supported in this browser.');
    }

    // Add window resize handler
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

/**
 * Handles the XR session's rendering loop.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 */
export function handleXRSession(renderer, scene, camera) {
    renderer.setAnimationLoop((timestamp, frame) => {
        if (frame) {
            try {
                // Get the XR session and reference space
                const session = renderer.xr.getSession();
                const refSpace = renderer.xr.getReferenceSpace();

                if (session && refSpace) {
                    // Get viewer pose
                    const pose = frame.getViewerPose(refSpace);
                    if (pose) {
                        // Handle pose data if needed
                        // pose.transform.position
                        // pose.transform.orientation
                    }

                    // Handle input sources
                    for (const source of session.inputSources) {
                        if (source.gamepad) {
                            handleGamepadInput(source.gamepad);
                        }
                        if (source.hand) {
                            handleHandInput(source.hand, frame, refSpace);
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
 */
function handleGamepadInput(gamepad) {
    if (!gamepad) return;

    // Handle buttons
    gamepad.buttons.forEach((button, index) => {
        if (button.pressed) {
            console.log(`XR Controller button ${index} pressed`);
            // Handle button press
        }
    });

    // Handle axes
    if (gamepad.axes.length >= 2) {
        const [x, y] = gamepad.axes;
        if (Math.abs(x) > 0.1 || Math.abs(y) > 0.1) {
            // Handle thumbstick/touchpad movement
            console.log(`XR Controller axes: x=${x}, y=${y}`);
        }
    }
}

/**
 * Handles hand tracking input in XR.
 * @param {XRHand} hand - The XR hand object.
 * @param {XRFrame} frame - The current XR frame.
 * @param {XRReferenceSpace} refSpace - The XR reference space.
 */
function handleHandInput(hand, frame, refSpace) {
    if (!hand || !frame || !refSpace) return;

    try {
        // Get joint poses
        for (const joint of hand.values()) {
            const pose = frame.getJointPose(joint, refSpace);
            if (pose) {
                // Handle joint pose data
                // pose.transform.position
                // pose.transform.orientation
            }
        }

        // Check for specific gestures
        const indexTip = hand.get('index-finger-tip');
        const thumbTip = hand.get('thumb-tip');
        
        if (indexTip && thumbTip) {
            const indexPose = frame.getJointPose(indexTip, refSpace);
            const thumbPose = frame.getJointPose(thumbTip, refSpace);
            
            if (indexPose && thumbPose) {
                // Calculate distance between index and thumb tips
                const distance = calculateDistance(
                    indexPose.transform.position,
                    thumbPose.transform.position
                );
                
                // Detect pinch gesture (2cm threshold)
                if (distance < 0.02) {
                    console.log('Pinch gesture detected');
                    // Handle pinch gesture
                }
            }
        }
    } catch (error) {
        console.error('Error handling hand input:', error);
    }
}

/**
 * Calculates distance between two XR positions.
 * @param {XRRigidTransform} pos1 - First position.
 * @param {XRRigidTransform} pos2 - Second position.
 * @returns {number} Distance between positions.
 */
function calculateDistance(pos1, pos2) {
    const dx = pos1.x - pos2.x;
    const dy = pos1.y - pos2.y;
    const dz = pos1.z - pos2.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
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
                // For example, updating controller positions, handling gestures, etc.
            }
        } catch (error) {
            console.error('Error updating XR frame:', error);
        }
    }
    renderer.render(scene, camera);
}
