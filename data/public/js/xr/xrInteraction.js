import * as THREE from 'three';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';

// Create a web panel for displaying node content
const webPanelGeometry = new THREE.PlaneGeometry(2, 1.5); // 2 meters wide, 1.5 meters tall
let webPanel = null;
let webPanelTexture = null;

/**
 * Creates a web panel in VR space
 * @param {string} url - The URL to display in the panel
 * @param {THREE.Scene} scene - The Three.js scene
 * @param {THREE.Vector3} position - The position to place the panel
 */
function createWebPanel(url, scene, position) {
    if (!url || !scene || !position) {
        console.error('Missing required parameters for createWebPanel');
        return;
    }

    // Remove existing panel if it exists
    if (webPanel) {
        if (webPanelTexture) {
            webPanelTexture.dispose();
        }
        scene.remove(webPanel);
    }

    // Create iframe to capture web content
    const iframe = document.createElement('iframe');
    iframe.style.width = '1024px';
    iframe.style.height = '768px';
    iframe.style.position = 'absolute';
    iframe.style.left = '-9999px'; // Hide iframe from view
    iframe.src = url;
    document.body.appendChild(iframe);

    // Create canvas to render iframe content
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 768;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Could not get 2D context for web panel canvas');
        return;
    }

    // Wait for iframe to load
    iframe.onload = () => {
        // Create texture from canvas
        webPanelTexture = new THREE.CanvasTexture(canvas);
        const material = new THREE.MeshBasicMaterial({ 
            map: webPanelTexture,
            side: THREE.DoubleSide
        });

        // Create panel mesh
        webPanel = new THREE.Mesh(webPanelGeometry, material);
        webPanel.position.copy(position);
        
        // Orient panel to face user
        webPanel.lookAt(0, position.y, 0);
        
        scene.add(webPanel);

        // Update texture periodically
        const updateInterval = setInterval(() => {
            if (!webPanel || !webPanelTexture) {
                clearInterval(updateInterval);
                return;
            }

            try {
                ctx.fillStyle = '#FFFFFF';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(iframe, 0, 0, canvas.width, canvas.height);
                webPanelTexture.needsUpdate = true;
            } catch (error) {
                console.error('Error updating web panel:', error);
                clearInterval(updateInterval);
            }
        }, 1000 / 30); // 30 FPS update rate

        // Store interval ID on the panel for cleanup
        webPanel.userData.updateInterval = updateInterval;
    };

    // Clean up iframe after a short delay
    setTimeout(() => {
        if (document.body.contains(iframe)) {
            document.body.removeChild(iframe);
        }
    }, 100);
}

/**
 * Initializes XR controller and hand interactions.
 * @param {THREE.Scene} scene - The Three.js scene.
 * @param {THREE.Camera} camera - The Three.js camera.
 * @param {THREE.WebGLRenderer} renderer - The Three.js renderer.
 * @param {Function} onSelect - Callback function for selection events.
 */
export function initXRInteraction(scene, camera, renderer, onSelect) {
    if (!scene || !camera || !renderer) {
        console.error('Missing required parameters for initXRInteraction');
        return;
    }

    // Store XR session globally for haptic feedback
    renderer.xr.addEventListener('sessionstart', (event) => {
        window.xrSession = event.target.getSession();
    });
    renderer.xr.addEventListener('sessionend', () => {
        window.xrSession = null;
        // Clean up web panel if it exists
        if (webPanel) {
            if (webPanel.userData.updateInterval) {
                clearInterval(webPanel.userData.updateInterval);
            }
            if (webPanelTexture) {
                webPanelTexture.dispose();
            }
            scene.remove(webPanel);
            webPanel = null;
            webPanelTexture = null;
        }
    });

    // Initialize controller model factories
    const controllerModelFactory = new XRControllerModelFactory();
    const handModelFactory = new XRHandModelFactory();

    // Set up controllers
    const controllers = setupControllers(scene, renderer, controllerModelFactory, (event) => {
        if (!event || !event.target) return;

        // Handle controller select event
        const controller = event.target;
        if (!controller.matrixWorld) return;

        const ray = new THREE.Vector3(0, 0, -1).applyMatrix4(controller.matrixWorld);
        const raycaster = new THREE.Raycaster();
        raycaster.set(controller.getWorldPosition(new THREE.Vector3()), ray);

        // Create a mock event for the node manager
        const mockEvent = {
            type: 'xr-select',
            detail: {
                controller,
                intersection: null
            }
        };

        // Find intersections
        const intersects = raycaster.intersectObjects(scene.children, true);
        if (intersects.length > 0) {
            mockEvent.detail.intersection = intersects[0];
            
            // Check if the intersected object is a node
            const nodeData = scene.nodeData?.get(intersects[0].object.uuid);
            if (nodeData) {
                // Get node URL
                const url = formatNodeNameToUrl(nodeData.label || nodeData.id);
                
                // Calculate position for web panel
                const panelPosition = new THREE.Vector3();
                panelPosition.copy(intersects[0].point);
                panelPosition.add(ray.multiplyScalar(2)); // Place panel 2 meters in front of hit point
                
                // Create web panel
                createWebPanel(url, scene, panelPosition);
            }

            if (onSelect) {
                onSelect(mockEvent);
            }

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

    // Update controller rays and handle interactions
    renderer.setAnimationLoop((timestamp, frame) => {
        if (!frame) return;

        // Update controller rays
        controllers.forEach((controller, i) => {
            const ray = controllerRays[i];
            if (ray && controller && controller.matrixWorld) {
                ray.position.setFromMatrixPosition(controller.matrixWorld);
                ray.quaternion.setFromRotationMatrix(controller.matrixWorld);
            }
        });

        // Handle XR input
        handleXRInput(frame, renderer.xr.getReferenceSpace(), controllers, hands, scene, onSelect);

        // Update labels to face camera
        xrLabelManager.update();

        // Update web panel to face user if it exists
        if (webPanel && camera) {
            const cameraPosition = new THREE.Vector3();
            camera.getWorldPosition(cameraPosition);
            cameraPosition.y = webPanel.position.y; // Keep panel vertical
            webPanel.lookAt(cameraPosition);
        }
    });

    return { controllers, hands, controllerRays, xrLabelManager };
}

/**
 * Format node name to URL
 */
function formatNodeNameToUrl(nodeName) {
    const baseUrl = window.location.origin;
    const formattedName = nodeName.toLowerCase().replace(/ /g, '-');
    return `${baseUrl}/#/page/${formattedName}`;
}

/**
 * Sets up XR controllers
 */
function setupControllers(scene, renderer, modelFactory, onSelect) {
    const controllers = [];
    
    for (let i = 0; i < 2; i++) {
        const controller = renderer.xr.getController(i);
        if (controller) {
            controller.addEventListener('select', onSelect);
            controller.addEventListener('connected', (event) => {
                if (event.data) {
                    controller.add(buildController(event.data));
                    controller.gamepad = event.data.gamepad;
                }
            });
            controller.addEventListener('disconnected', () => {
                if (controller.children.length > 0) {
                    controller.remove(controller.children[0]);
                }
                controller.gamepad = null;
            });
            scene.add(controller);

            const grip = renderer.xr.getControllerGrip(i);
            if (grip) {
                grip.add(modelFactory.createControllerModel(grip));
                scene.add(grip);
            }

            controllers.push(controller);
        }
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
        if (hand) {
            // Add hand model
            const model = modelFactory.createHandModel(hand, 'mesh');
            hand.add(model);
            
            // Setup hand joints
            hand.addEventListener('connected', (event) => {
                if (event.data) {
                    setupHandJoints(hand, event.data);
                }
            });

            scene.add(hand);
            hands.push(hand);
        }
    }

    return hands;
}

/**
 * Creates visual rays for controllers
 */
function createControllerRays(controllers) {
    return controllers.map(() => {
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
}

/**
 * Builds controller visual representation
 */
function buildController(data) {
    if (!data || !data.targetRayMode) return null;

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

        default:
            return null;
    }
}

/**
 * Sets up hand joints for tracking
 */
function setupHandJoints(hand, data) {
    if (!hand || !data || !data.joints) return;

    const joints = {};
    
    // Create spheres for each joint
    for (const jointName in data.joints) {
        if (data.joints.hasOwnProperty(jointName)) {
            const joint = data.joints[jointName];
            if (joint) {
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
        }
    }

    hand.joints = joints;
}

/**
 * Handles XR input (controllers and hands)
 */
export function handleXRInput(frame, referenceSpace, controllers, hands, scene, onSelect) {
    if (!frame || !referenceSpace) return;

    // Handle controller input
    controllers.forEach((controller) => {
        if (!controller) return;

        const ray = controller.getWorldDirection(new THREE.Vector3());
        const raycaster = new THREE.Raycaster();
        raycaster.set(controller.position, ray);

        const intersects = raycaster.intersectObjects(scene.children, true);
        if (intersects.length > 0) {
            // Create mock event
            const mockEvent = {
                type: 'xr-select',
                detail: {
                    controller,
                    intersection: intersects[0]
                }
            };

            if (onSelect) {
                onSelect(mockEvent);
            }
        }
    });

    // Handle hand tracking
    hands.forEach(hand => {
        if (!hand || !hand.joints) return;

        const indexTip = hand.joints['index-finger-tip'];
        const thumbTip = hand.joints['thumb-tip'];

        if (indexTip && thumbTip) {
            // Detect pinch gesture
            const distance = indexTip.position.distanceTo(thumbTip.position);
            if (distance < 0.02) {  // 2cm threshold for pinch
                const position = indexTip.position.clone().add(thumbTip.position).multiplyScalar(0.5);
                const direction = new THREE.Vector3().subVectors(indexTip.position, thumbTip.position).normalize();
                
                const raycaster = new THREE.Raycaster();
                raycaster.set(position, direction);

                const intersects = raycaster.intersectObjects(scene.children, true);
                if (intersects.length > 0) {
                    // Create mock event
                    const mockEvent = {
                        type: 'xr-select',
                        detail: {
                            hand,
                            intersection: intersects[0]
                        }
                    };

                    if (onSelect) {
                        onSelect(mockEvent);
                    }
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
        if (!text || !position) return;

        // Remove existing label if present
        this.hideLabel(text);

        // Create label sprite
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (!context) return;
        
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
            if (label.material.map) {
                label.material.map.dispose();
            }
            if (label.material) {
                label.material.dispose();
            }
            this.labels.delete(text);
        }
    }

    update() {
        if (!this.camera) return;

        // Update all labels to face camera
        this.labels.forEach(label => {
            if (label) {
                label.lookAt(this.camera.position);
            }
        });
    }

    dispose() {
        // Clean up all labels
        this.labels.forEach((label, text) => {
            this.hideLabel(text);
        });
        this.labels.clear();
    }
}
