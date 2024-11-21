import * as THREE from 'three';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';

/**
 * Initializes XR controller and hand interactions.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.Camera} camera - The Three.js camera.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {NodeManager} nodeManager - The node manager instance.
 */
export function initXRInteraction(scene, camera, renderer, nodeManager) {
    // Store XR session globally for haptic feedback
    renderer.xr.addEventListener('sessionstart', (event) => {
        window.xrSession = event.target.getSession();
        nodeManager.xrEnabled = true;
    });
    renderer.xr.addEventListener('sessionend', () => {
        window.xrSession = null;
        nodeManager.xrEnabled = false;
    });

    // Initialize controller model factories
    const controllerModelFactory = new XRControllerModelFactory();
    const handModelFactory = new XRHandModelFactory();

    // Set up controllers with node manager
    const controllers = setupControllers(scene, renderer, controllerModelFactory, (event) => {
        // Handle controller select event
        const controller = event.target;
        const ray = controller.getWorldDirection(new THREE.Vector3());
        const raycaster = new THREE.Raycaster();
        raycaster.set(controller.position, ray);

        const intersects = raycaster.intersectObjects(Array.from(nodeManager.nodeMeshes.values()));
        if (intersects.length > 0) {
            nodeManager.handleClick(null, true, intersects[0].object);

            // Trigger haptic feedback
            const gamepad = controller.gamepad;
            if (gamepad?.hapticActuators?.length > 0) {
                gamepad.hapticActuators[0].pulse(0.5, 100);
            }
        }
    });
    
    // Set up hands
    const hands = setupHands(scene, renderer, handModelFactory);

    // Create interaction rays
    const controllerRays = createControllerRays(controllers);
    scene.add(...controllerRays);

    // Create XR label manager
    const xrLabelManager = new XRLabelManager(scene, camera);
    nodeManager.xrLabelManager = xrLabelManager;

    // Update controller rays and handle interactions
    renderer.setAnimationLoop((timestamp, frame) => {
        if (!frame) return;

        // Update controller rays
        controllers.forEach((controller, i) => {
            const ray = controllerRays[i];
            ray.position.copy(controller.position);
            ray.quaternion.copy(controller.quaternion);
        });

        // Handle XR input
        handleXRInput(frame, renderer.xr.getReferenceSpace(), controllers, hands, nodeManager);

        // Update labels to face camera
        xrLabelManager.update();
    });

    return { controllers, hands, controllerRays, xrLabelManager };
}

/**
 * Sets up XR controllers
 */
function setupControllers(scene, renderer, modelFactory, onSelect) {
    const controllers = [];
    
    for (let i = 0; i < 2; i++) {
        const controller = renderer.xr.getController(i);
        controller.addEventListener('select', onSelect);
        controller.addEventListener('connected', (event) => {
            controller.add(buildController(event.data));
            controller.gamepad = event.data.gamepad;
        });
        controller.addEventListener('disconnected', () => {
            controller.remove(controller.children[0]);
            controller.gamepad = null;
        });
        scene.add(controller);

        const grip = renderer.xr.getControllerGrip(i);
        grip.add(modelFactory.createControllerModel(grip));
        scene.add(grip);

        controllers.push(controller);
    }

    return controllers;
}

/**
 * Sets up hand tracking
 */
function setupHands(scene, renderer, modelFactory) {
    const hands = [];

    for (let i = 0; i < 2; i++) {
        const hand = renderer.xr.getHand(i);
        
        // Add hand model
        const model = modelFactory.createHandModel(hand, 'mesh');
        hand.add(model);
        
        // Setup hand joints
        hand.addEventListener('connected', (event) => {
            setupHandJoints(hand, event.data);
        });

        scene.add(hand);
        hands.push(hand);
    }

    return hands;
}

/**
 * Creates visual rays for controllers
 */
function createControllerRays(controllers) {
    const rays = controllers.map(() => {
        const geometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 0, -5)  // 5 meter ray
        ]);
        const material = new THREE.LineBasicMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.5
        });
        return new THREE.Line(geometry, material);
    });

    return rays;
}

/**
 * Builds controller visual representation
 */
function buildController(data) {
    let geometry, material;

    switch (data.targetRayMode) {
        case 'tracked-pointer':
            geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, -1], 3));
            material = new THREE.LineBasicMaterial({
                color: 0x00ff00,
                transparent: true,
                opacity: 0.5
            });
            return new THREE.Line(geometry, material);

        case 'gaze':
            geometry = new THREE.RingGeometry(0.02, 0.04, 32).translate(0, 0, -1);
            material = new THREE.MeshBasicMaterial({
                color: 0x00ff00,
                opacity: 0.5,
                transparent: true
            });
            return new THREE.Mesh(geometry, material);
    }
}

/**
 * Sets up hand joints for tracking
 */
function setupHandJoints(hand, data) {
    const joints = {};
    
    // Create spheres for each joint
    for (const jointName in data.joints) {
        const joint = data.joints[jointName];
        const geometry = new THREE.SphereGeometry(0.008);
        const material = new THREE.MeshPhongMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.5
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.visible = false;
        
        joints[jointName] = mesh;
        hand.add(mesh);
    }

    hand.joints = joints;
}

/**
 * Handles XR input (controllers and hands)
 */
export function handleXRInput(frame, referenceSpace, controllers, hands, nodeManager) {
    if (!frame) return;

    // Handle controller input
    controllers.forEach((controller) => {
        const ray = controller.getWorldDirection(new THREE.Vector3());
        const raycaster = new THREE.Raycaster();
        raycaster.set(controller.position, ray);

        const intersects = raycaster.intersectObjects(Array.from(nodeManager.nodeMeshes.values()));
        if (intersects.length > 0) {
            // Highlight intersected node
            const mesh = intersects[0].object;
            const originalEmissive = mesh.material.emissiveIntensity;
            mesh.material.emissiveIntensity = 2.0;
            setTimeout(() => {
                mesh.material.emissiveIntensity = originalEmissive;
            }, 100);

            // Show label in XR
            if (nodeManager.xrLabelManager) {
                const nodeId = Array.from(nodeManager.nodeMeshes.entries())
                    .find(([_, m]) => m === mesh)?.[0];
                if (nodeId) {
                    const nodeData = nodeManager.nodeData.get(nodeId);
                    if (nodeData) {
                        nodeManager.xrLabelManager.showLabel(nodeData.label || nodeId, mesh.position);
                    }
                }
            }
        }
    });

    // Handle hand tracking
    hands.forEach(hand => {
        if (hand.joints['index-finger-tip'] && hand.joints['thumb-tip']) {
            const indexTip = hand.joints['index-finger-tip'];
            const thumbTip = hand.joints['thumb-tip'];

            // Detect pinch gesture
            const distance = indexTip.position.distanceTo(thumbTip.position);
            if (distance < 0.02) {  // 2cm threshold for pinch
                const position = indexTip.position.clone().add(thumbTip.position).multiplyScalar(0.5);
                const direction = new THREE.Vector3().subVectors(indexTip.position, thumbTip.position).normalize();
                
                const raycaster = new THREE.Raycaster();
                raycaster.set(position, direction);

                const intersects = raycaster.intersectObjects(Array.from(nodeManager.nodeMeshes.values()));
                if (intersects.length > 0) {
                    nodeManager.handleClick(null, true, intersects[0].object);
                }
            }
        }
    });
}

/**
 * Creates and manages labels in XR space
 */
export class XRLabelManager {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.labels = new Map();
        this.labelTimeout = 2000;  // Labels disappear after 2 seconds
    }

    showLabel(text, position, options = {}) {
        // Remove existing label if present
        this.hideLabel(text);

        // Create label sprite
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        // Configure canvas
        canvas.width = 256;
        canvas.height = 128;
        
        // Draw background
        context.fillStyle = options.backgroundColor || 'rgba(0, 0, 0, 0.8)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw text
        context.font = options.font || '24px Arial';
        context.fillStyle = options.color || '#ffffff';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, canvas.width / 2, canvas.height / 2);

        // Create sprite
        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ 
            map: texture,
            transparent: true,
            opacity: 0.8
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        
        // Position and scale sprite
        sprite.position.copy(position);
        sprite.position.y += 0.1;  // Offset slightly above node
        sprite.scale.set(0.2, 0.1, 1);

        this.scene.add(sprite);
        this.labels.set(text, sprite);

        // Remove label after timeout
        setTimeout(() => this.hideLabel(text), this.labelTimeout);
    }

    hideLabel(text) {
        const label = this.labels.get(text);
        if (label) {
            this.scene.remove(label);
            if (label.material.map) label.material.map.dispose();
            label.material.dispose();
            this.labels.delete(text);
        }
    }

    update() {
        // Update all labels to face camera
        this.labels.forEach(label => {
            label.lookAt(this.camera.position);
        });
    }

    dispose() {
        // Clean up all labels
        this.labels.forEach((label, text) => {
            this.hideLabel(text);
        });
    }
}
