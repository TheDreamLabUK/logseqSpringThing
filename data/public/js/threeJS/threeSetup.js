// public/js/threeJS/threeSetup.js

import * as THREE from 'three';

/**
 * Initializes the Three.js scene, camera, and renderer.
 * @returns {object} An object containing the scene, camera, and renderer.
 */
export function initThreeScene() {
    // Create the scene
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.002);

    // Create the camera
    const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.set(0, 0, 100);

    // Create the renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.xr.enabled = true; // Enable WebXR

    // Append the renderer to the DOM
    document.getElementById('scene-container').appendChild(renderer.domElement);

    return { scene, camera, renderer };
}

/**
 * Creates and configures orbit controls for the camera.
 * @param {THREE.Camera} camera - The Three.js camera.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @returns {OrbitControls} The configured orbit controls.
 */
export function createOrbitControls(camera, renderer) {
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    return controls;
}

/**
 * Handles window resize events by updating the camera and renderer.
 * @param {THREE.Camera} camera - The Three.js camera.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 */
export function updateSceneSize(camera, renderer) {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
}
