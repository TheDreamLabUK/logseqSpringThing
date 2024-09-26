// public/js/xr/xrInteraction.js

import * as THREE from 'three';

/**
 * Initializes XR controller interactions.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.Camera} camera - The Three.js camera.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {function} onSelect - Callback function when an object is selected.
 */
export function initXRInteraction(scene, camera, renderer, onSelect) {
    const controller1 = renderer.xr.getController(0);
    const controller2 = renderer.xr.getController(1);

    controller1.addEventListener('select', onSelect);
    controller2.addEventListener('select', onSelect);

    scene.add(controller1);
    scene.add(controller2);

    // Optional: Add visual indicators for controllers
    const controllerModelFactory = new THREE.XRControllerModelFactory();

    const controllerGrip1 = renderer.xr.getControllerGrip(0);
    controllerGrip1.add(controllerModelFactory.createControllerModel(controllerGrip1));
    scene.add(controllerGrip1);

    const controllerGrip2 = renderer.xr.getControllerGrip(1);
    controllerGrip2.add(controllerModelFactory.createControllerModel(controllerGrip2));
    scene.add(controllerGrip2);
}

/**
 * Handles controller selection events.
 * @param {THREE.Intersection[]} intersects - Array of intersected objects.
 * @param {function} onSelect - Callback function to handle selection.
 */
export function handleControllerSelection(intersects, onSelect) {
    if (intersects.length > 0) {
        const selectedObject = intersects[0].object;
        onSelect(selectedObject);
    }
}

/**
 * Adds labels as billboards to nodes.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.Camera} camera - The Three.js camera.
 * @param {Array} nodes - Array of node objects.
 */
export function addNodeLabels(scene, camera, nodes) {
    const loader = new THREE.FontLoader();

    loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', function (font) {
        nodes.forEach(node => {
            const textGeometry = new THREE.TextGeometry(node.name, {
                font: font,
                size: 1,
                height: 0.1,
                curveSegments: 12,
                bevelEnabled: false,
            });

            const textMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
            const textMesh = new THREE.Mesh(textGeometry, textMaterial);

            // Position the label above the node
            textMesh.position.set(node.x, node.y + 3, node.z);
            textMesh.lookAt(camera.position); // Make the label face the camera

            scene.add(textMesh);
            node.labelMesh = textMesh; // Store reference for updates
        });
    });
}

/**
 * Updates label orientations to always face the camera.
 * @param {THREE.Camera} camera - The Three.js camera.
 * @param {Array} nodes - Array of node objects with label meshes.
 */
export function updateLabelOrientations(camera, nodes) {
    nodes.forEach(node => {
        if (node.labelMesh) {
            node.labelMesh.lookAt(camera.position);
        }
    });
}
