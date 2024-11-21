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
        // Check for both VR and AR support
        Promise.all([
            navigator.xr.isSessionSupported('immersive-vr'),
            navigator.xr.isSessionSupported('immersive-ar')
        ]).then(([vrSupported, arSupported]) => {
            if (vrSupported || arSupported) {
                // Session initialization settings
                const sessionInit = {
                    requiredFeatures: ['hand-tracking'], // Enable hand tracking
                    optionalFeatures: ['local-floor', 'bounded-floor']
                };

                // Create and configure XR button
                const xrButton = XRButton.createButton(renderer, {
                    ...sessionInit,
                    mode: arSupported ? 'immersive-ar' : 'immersive-vr', // Prefer AR if available
                    sessionInit: sessionInit
                });

                document.body.appendChild(xrButton);

                // Set up session start event handler
                renderer.xr.addEventListener('sessionstart', () => {
                    console.log('XR session started');
                    // Additional session start setup can go here
                });

                // Set up session end event handler
                renderer.xr.addEventListener('sessionend', () => {
                    console.log('XR session ended');
                    // Additional cleanup can go here
                });
            }
        }).catch((err) => {
            console.error('Error checking XR session support:', err);
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
            // Handle XR input sources (controllers/hands)
            const session = renderer.xr.getSession();
            if (session) {
                for (const source of session.inputSources) {
                    if (source.hand) {
                        // Handle hand tracking data
                        handleHandInput(source.hand);
                    }
                }
            }
        }
        renderer.render(scene, camera);
    });
}

/**
 * Updates the XR frame with hand tracking data.
 * @param {XRHand} hand - The XR hand object containing joint data.
 */
function handleHandInput(hand) {
    // Process hand tracking data
    for (const joint of hand.values()) {
        if (joint.jointName) {
            // You can access joint positions and rotations here
            // joint.pose.transform.position
            // joint.pose.transform.orientation
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
        const session = renderer.xr.getSession();
        if (session) {
            // Additional frame updates can be handled here
            // For example, updating controller positions, handling gestures, etc.
        }
    }
    renderer.render(scene, camera);
}
