// public/js/xr/xrSetup.js

import * as THREE from 'three';

/**
 * Initializes the WebXR session for immersive experiences.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 */
export function initXRSession(renderer, scene, camera) {
    if ('xr' in navigator) {
        navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
            if (supported) {
                const enterXRButton = document.createElement('button');
                enterXRButton.id = 'enter-vr-button';
                enterXRButton.textContent = 'Enter VR';
                enterXRButton.style.position = 'absolute';
                enterXRButton.style.top = '20px';
                enterXRButton.style.right = '20px';
                enterXRButton.style.padding = '10px 15px';
                enterXRButton.style.backgroundColor = 'rgba(0, 0, 0, 0.6)';
                enterXRButton.style.color = 'white';
                enterXRButton.style.border = 'none';
                enterXRButton.style.borderRadius = '4px';
                enterXRButton.style.cursor = 'pointer';
                enterXRButton.style.transition = 'background-color 0.3s';

                enterXRButton.addEventListener('mouseenter', () => {
                    enterXRButton.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                });

                enterXRButton.addEventListener('mouseleave', () => {
                    enterXRButton.style.backgroundColor = 'rgba(0, 0, 0, 0.6)';
                });

                enterXRButton.onclick = () => {
                    renderer.xr.setReferenceSpaceType('local');
                    navigator.xr.requestSession('immersive-vr').then((session) => {
                        renderer.xr.setSession(session);
                    });
                };

                document.body.appendChild(enterXRButton);
            }
        }).catch((err) => {
            console.error('Error checking XR session support:', err);
        });
    } else {
        console.warn('WebXR not supported in this browser.');
    }
}

/**
 * Handles the XR session's rendering loop.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 */
export function handleXRSession(renderer, scene, camera) {
    renderer.setAnimationLoop(() => {
        renderer.render(scene, camera);
    });
}

/**
 * Updates the XR frame, if necessary.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.PerspectiveCamera} camera - The Three.js camera.
 */
export function updateXRFrame(renderer, scene, camera) {
    // Additional updates can be handled here if needed
    renderer.render(scene, camera);
}
