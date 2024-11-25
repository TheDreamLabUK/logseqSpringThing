// public/js/threeJS/threeSetup.js

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Initializes the Three.js scene, camera, and renderer.
 * @returns {object} An object containing the scene, camera, and renderer.
 */
export function initThreeScene() {
    // Create the scene
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.002);

    // Create the camera with XR-friendly settings
    const camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,  // Closer near plane for XR
        1000
    );
    camera.position.set(0, 1.6, 3); // Default height ~1.6m (average human height)

    // Try WebGL2 first, fall back to WebGL1
    let renderer;
    try {
        const canvas = document.createElement('canvas');
        const contextAttributes = {
            alpha: true,
            antialias: true,
            powerPreference: "high-performance",
            failIfMajorPerformanceCaveat: false,
            preserveDrawingBuffer: false
        };

        // Try WebGL2 first
        let gl = canvas.getContext('webgl2', contextAttributes);
        let isWebGL2 = !!gl;

        if (!gl) {
            console.warn('WebGL2 not available, falling back to WebGL1');
            gl = canvas.getContext('webgl', contextAttributes) ||
                 canvas.getContext('experimental-webgl', contextAttributes);
            isWebGL2 = false;
        }

        if (!gl) {
            throw new Error('WebGL not supported');
        }

        // Create renderer with detected context
        renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            context: gl,
            antialias: true,
            alpha: true,
            powerPreference: "high-performance",
            preserveDrawingBuffer: false
        });

        // Configure renderer based on WebGL version
        if (isWebGL2) {
            console.log('Using WebGL2 renderer');
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        } else {
            console.log('Using WebGL1 renderer');
            renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
            renderer.shadowMap.type = THREE.PCFShadowMap;
        }

        // Common renderer settings
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.xr.enabled = true;

        // Store WebGL version for other components
        renderer.capabilities.isWebGL2 = isWebGL2;

    } catch (error) {
        console.error('Error creating renderer:', error);
        throw error;
    }

    // Add WebGL context loss handling
    renderer.domElement.addEventListener('webglcontextlost', handleContextLost, false);
    renderer.domElement.addEventListener('webglcontextrestored', handleContextRestored, false);

    // Set up XR-friendly lighting
    setupLighting(scene);

    // Append the renderer to the DOM
    const container = document.getElementById('scene-container');
    if (container) {
        container.appendChild(renderer.domElement);
    } else {
        document.body.appendChild(renderer.domElement);
    }

    // Add ambient light for better visibility in XR
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    // Set texture memory hints
    THREE.TextureLoader.prototype.crossOrigin = 'anonymous';
    renderer.info.autoReset = true;
    renderer.info.reset();

    return { scene, camera, renderer };
}

/**
 * Handle WebGL context loss
 * @param {Event} event - The context loss event
 */
function handleContextLost(event) {
    event.preventDefault();
    console.warn('WebGL context lost. Attempting to restore...');
}

/**
 * Handle WebGL context restoration
 */
function handleContextRestored() {
    console.log('WebGL context restored');
    // Force a full scene refresh
    window.dispatchEvent(new Event('webglcontextrestored'));
}

/**
 * Sets up XR-friendly lighting in the scene
 * @param {THREE.Scene} scene - The Three.js scene
 */
function setupLighting(scene) {
    // Main directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 5, 5);
    directionalLight.castShadow = true;
    
    // Optimize shadow map settings
    directionalLight.shadow.mapSize.width = 1024; // Reduced for WebGL1
    directionalLight.shadow.mapSize.height = 1024;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 500;
    directionalLight.shadow.bias = -0.0001;
    
    scene.add(directionalLight);

    // Fill light
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
    fillLight.position.set(-5, 5, -5);
    scene.add(fillLight);

    // Ambient light for overall scene brightness
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
}

/**
 * Creates and configures orbit controls for the camera.
 * @param {THREE.Camera} camera - The Three.js camera.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @returns {OrbitControls} The configured orbit controls.
 */
export function createOrbitControls(camera, renderer) {
    const controls = new OrbitControls(camera, renderer.domElement);
    
    // Configure controls for XR compatibility
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI * 0.95;
    controls.minDistance = 1;
    controls.maxDistance = 50;
    controls.enablePan = true;
    controls.panSpeed = 0.5;
    controls.rotateSpeed = 0.5;
    controls.zoomSpeed = 0.5;
    
    // Disable controls when in XR mode
    renderer.xr.addEventListener('sessionstart', () => {
        controls.enabled = false;
    });
    
    renderer.xr.addEventListener('sessionend', () => {
        controls.enabled = true;
    });

    return controls;
}

/**
 * Handles window resize events by updating the camera and renderer.
 * @param {THREE.Camera} camera - The Three.js camera.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 */
export function updateSceneSize(camera, renderer) {
    // Only update if not in XR session
    if (!renderer.xr.isPresenting) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

/**
 * Creates a basic environment for XR scenes
 * @param {THREE.Scene} scene - The Three.js scene
 * @returns {Object} Environment meshes for cleanup
 */
export function createBasicEnvironment(scene) {
    // Add a ground plane for reference and shadows
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshStandardMaterial({ 
        color: 0x808080,
        roughness: 0.8,
        metalness: 0.2,
        transparent: true,
        opacity: 0.8
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // Add grid helper for spatial reference
    const gridHelper = new THREE.GridHelper(100, 100);
    gridHelper.material.transparent = true;
    gridHelper.material.opacity = 0.2;
    scene.add(gridHelper);

    return {
        ground,
        gridHelper,
        dispose: () => {
            groundGeometry.dispose();
            groundMaterial.dispose();
            if (gridHelper.material) {
                gridHelper.material.dispose();
            }
            if (gridHelper.geometry) {
                gridHelper.geometry.dispose();
            }
            scene.remove(ground);
            scene.remove(gridHelper);
        }
    };
}

/**
 * Disposes of Three.js resources
 * @param {THREE.Scene} scene - The Three.js scene
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer
 * @param {OrbitControls} controls - The orbit controls
 */
export function disposeThreeResources(scene, renderer, controls) {
    // Dispose of scene objects
    scene.traverse((object) => {
        if (object.geometry) {
            object.geometry.dispose();
        }
        
        if (object.material) {
            if (Array.isArray(object.material)) {
                object.material.forEach(material => {
                    if (material.map) material.map.dispose();
                    material.dispose();
                });
            } else {
                if (object.material.map) object.material.map.dispose();
                object.material.dispose();
            }
        }
    });

    // Dispose of renderer
    renderer.dispose();
    renderer.forceContextLoss();
    renderer.domElement.remove();

    // Remove context loss listeners
    renderer.domElement.removeEventListener('webglcontextlost', handleContextLost);
    renderer.domElement.removeEventListener('webglcontextrestored', handleContextRestored);

    // Dispose of controls
    if (controls) {
        controls.dispose();
    }
}
