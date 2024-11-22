import * as THREE from 'three';
import { XRHandModelFactory } from 'three/examples/jsm/webxr/XRHandModelFactory.js';

// Hand tracking setup
const handModelFactory = new XRHandModelFactory();
const hands = {
    left: null,
    right: null
};

// To store object grab states for both hands
const grabStates = {
    left: { grabbedObject: null, pinching: false },
    right: { grabbedObject: null, pinching: false }
};

// Visual feedback for pinch state
const pinchIndicators = {
    left: null,
    right: null
};

// Collection of interactable objects
const interactableObjects = new Set();

// Constants for interaction
const PINCH_THRESHOLD = 0.015; // Smaller threshold for more precise detection
const GRAB_THRESHOLD = 0.08;   // Distance to grab objects
const PINCH_STRENGTH_THRESHOLD = 0.7; // Required strength for pinch

// XR Label Manager Class
export class XRLabelManager {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.labels = new Map();
    }

    createLabel(text, position) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 128;
        
        context.fillStyle = '#ffffff';
        context.font = '24px Arial';
        context.fillText(text, 10, 64);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ 
            map: texture,
            transparent: true,
            depthWrite: false
        });
        
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position);
        sprite.scale.set(0.5, 0.25, 1);
        
        this.scene.add(sprite);
        this.labels.set(text, sprite);
        
        return sprite;
    }

    updateLabel(text, position) {
        const label = this.labels.get(text);
        if (label) {
            label.position.copy(position);
            label.lookAt(this.camera.position);
        }
    }

    removeLabel(text) {
        const label = this.labels.get(text);
        if (label) {
            this.scene.remove(label);
            this.labels.delete(text);
        }
    }

    updateAll() {
        this.labels.forEach(label => {
            label.lookAt(this.camera.position);
        });
    }

    dispose() {
        this.labels.forEach(label => {
            if (label.material.map) {
                label.material.map.dispose();
            }
            label.material.dispose();
            this.scene.remove(label);
        });
        this.labels.clear();
    }
}

// Detect pinch with strength
function isPinching(hand) {
    const indexTip = hand.joints['index-finger-tip'];
    const thumbTip = hand.joints['thumb-tip'];

    if (indexTip && thumbTip) {
        const distance = indexTip.position.distanceTo(thumbTip.position);
        // Calculate pinch strength (1 when touching, 0 when far)
        const strength = Math.max(0, 1 - (distance / PINCH_THRESHOLD));
        return {
            isPinched: distance < PINCH_THRESHOLD,
            strength: strength
        };
    }
    return { isPinched: false, strength: 0 };
}

// Create visual feedback sphere for pinch state
function createPinchIndicator() {
    const geometry = new THREE.SphereGeometry(0.01);
    const material = new THREE.MeshPhongMaterial({
        color: 0x00ff00,
        transparent: true,
        opacity: 0.5
    });
    return new THREE.Mesh(geometry, material);
}

// Update pinch indicator position and appearance
function updatePinchIndicator(hand, indicator) {
    if (hand && hand.joints && indicator) {
        const indexTip = hand.joints['index-finger-tip'];
        const thumbTip = hand.joints['thumb-tip'];
        
        if (indexTip && thumbTip) {
            indicator.position.copy(indexTip.position).lerp(thumbTip.position, 0.5);
            const { strength } = isPinching(hand);
            indicator.material.opacity = strength * 0.8;
            indicator.scale.setScalar(1 - (strength * 0.5));
        }
    }
}

// Initialize XR interaction
export function initXRInteraction(scene, camera, renderer, onSelect) {
    const xrLabelManager = new XRLabelManager(scene, camera);
    
    // Create default interactable objects
    const createInteractableObject = (position) => {
        const geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
        const material = new THREE.MeshStandardMaterial({ 
            color: 0xff0000,
            roughness: 0.7,
            metalness: 0.3
        });
        const object = new THREE.Mesh(geometry, material);
        object.position.copy(position);
        object.userData.interactable = true;
        scene.add(object);
        interactableObjects.add(object);
        return object;
    };

    // Create multiple interactable objects
    createInteractableObject(new THREE.Vector3(0, 1.5, -1));
    createInteractableObject(new THREE.Vector3(0.2, 1.5, -1));
    createInteractableObject(new THREE.Vector3(-0.2, 1.5, -1));

    // Initialize pinch indicators
    pinchIndicators.left = createPinchIndicator();
    pinchIndicators.right = createPinchIndicator();
    scene.add(pinchIndicators.left);
    scene.add(pinchIndicators.right);

    // Set up hand tracking
    renderer.xr.addEventListener('sessionstart', () => {
        const session = renderer.xr.getSession();
        
        hands.left = renderer.xr.getHand(0);
        hands.right = renderer.xr.getHand(1);
        
        // Add hand models
        for (const [handedness, hand] of Object.entries(hands)) {
            if (hand) {
                const handModel = handModelFactory.createHandModel(hand, 'mesh');
                hand.add(handModel);
                scene.add(hand);
            }
        }
    });

    // Create update function
    const update = () => {
        // Update both hands
        for (const [handedness, hand] of Object.entries(hands)) {
            if (hand?.joints) {
                const grabState = grabStates[handedness];
                const { isPinched, strength } = isPinching(hand);
                
                // Update pinch indicator
                updatePinchIndicator(hand, pinchIndicators[handedness]);

                if (isPinched && strength > PINCH_STRENGTH_THRESHOLD) {
                    if (!grabState.grabbedObject) {
                        // Check for nearby interactable objects
                        const indexTip = hand.joints['index-finger-tip'];
                        
                        for (const object of interactableObjects) {
                            const distance = indexTip.position.distanceTo(object.position);
                            if (distance < GRAB_THRESHOLD && !object.userData.isGrabbed) {
                                grabState.grabbedObject = object;
                                object.userData.isGrabbed = true;
                                object.material.emissive.setHex(0x222222);
                                break;
                            }
                        }
                    } else if (grabState.grabbedObject) {
                        // Move grabbed object
                        const indexTip = hand.joints['index-finger-tip'];
                        grabState.grabbedObject.position.copy(indexTip.position);
                    }
                    grabState.pinching = true;
                } else if (grabState.pinching) {
                    // Release object
                    if (grabState.grabbedObject) {
                        grabState.grabbedObject.userData.isGrabbed = false;
                        grabState.grabbedObject.material.emissive.setHex(0x000000);
                        grabState.grabbedObject = null;
                    }
                    grabState.pinching = false;
                }
            }
        }

        // Update label orientations in XR
        if (renderer.xr.isPresenting) {
            const camera = renderer.xr.getCamera();
            scene.traverse((object) => {
                if (object.isSprite) {
                    object.lookAt(camera.position);
                }
            });
        }
    };

    return {
        hands: Object.values(hands),
        controllers: [], // For compatibility with existing code
        xrLabelManager,
        update,
        addInteractableObject: (object) => {
            object.userData.interactable = true;
            interactableObjects.add(object);
        },
        removeInteractableObject: (object) => {
            interactableObjects.delete(object);
        }
    };
}

// Handle XR input
export function handleXRInput(frame, referenceSpace) {
    // Update both hands
    for (const [handedness, hand] of Object.entries(hands)) {
        if (hand?.joints) {
            const grabState = grabStates[handedness];
            const { isPinched, strength } = isPinching(hand);
            
            // Update pinch indicator
            updatePinchIndicator(hand, pinchIndicators[handedness]);

            if (isPinched && strength > PINCH_STRENGTH_THRESHOLD) {
                if (!grabState.grabbedObject) {
                    // Check for nearby interactable objects
                    const indexTip = hand.joints['index-finger-tip'];
                    
                    for (const object of interactableObjects) {
                        const distance = indexTip.position.distanceTo(object.position);
                        if (distance < GRAB_THRESHOLD && !object.userData.isGrabbed) {
                            grabState.grabbedObject = object;
                            object.userData.isGrabbed = true;
                            object.material.emissive.setHex(0x222222);
                            break;
                        }
                    }
                } else if (grabState.grabbedObject) {
                    // Move grabbed object
                    const indexTip = hand.joints['index-finger-tip'];
                    grabState.grabbedObject.position.copy(indexTip.position);
                }
                grabState.pinching = true;
            } else if (grabState.pinching) {
                // Release object
                if (grabState.grabbedObject) {
                    grabState.grabbedObject.userData.isGrabbed = false;
                    grabState.grabbedObject.material.emissive.setHex(0x000000);
                    grabState.grabbedObject = null;
                }
                grabState.pinching = false;
            }
        }
    }
}
