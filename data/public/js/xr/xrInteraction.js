import * as THREE from 'three';
import { XRHandModelFactory } from 'three/examples/jsm/webxr/XRHandModelFactory.js';

// Constants for interaction
const PINCH_THRESHOLD = 0.015;
const GRAB_THRESHOLD = 0.08;
const PINCH_STRENGTH_THRESHOLD = 0.7;
const LABEL_SIZE = { width: 256, height: 128 };
const LABEL_SCALE = { x: 0.5, y: 0.25, z: 1 };

// Resource pools
const materialPool = new Map();
const geometryPool = new Map();
const texturePool = new Map();

// Hand tracking setup
const handModelFactory = new XRHandModelFactory();
const hands = { left: null, right: null };
const grabStates = {
    left: { grabbedObject: null, pinching: false },
    right: { grabbedObject: null, pinching: false }
};
const pinchIndicators = { left: null, right: null };
const interactableObjects = new Set();

// XR Label Manager Class
export class XRLabelManager {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.labels = new Map();
        this.labelCanvas = document.createElement('canvas');
        this.labelContext = this.labelCanvas.getContext('2d', {
            alpha: true,
            desynchronized: true
        });
        
        // Set canvas size to power of 2
        this.labelCanvas.width = LABEL_SIZE.width;
        this.labelCanvas.height = LABEL_SIZE.height;
    }

    /**
     * Get or create a texture for label
     * @param {string} text - Label text
     * @returns {THREE.Texture} The texture
     */
    getTexture(text) {
        if (texturePool.has(text)) {
            return texturePool.get(text);
        }

        // Clear canvas
        this.labelContext.clearRect(0, 0, LABEL_SIZE.width, LABEL_SIZE.height);
        
        // Draw background
        this.labelContext.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.labelContext.fillRect(0, 0, LABEL_SIZE.width, LABEL_SIZE.height);
        
        // Draw text
        this.labelContext.fillStyle = '#ffffff';
        this.labelContext.font = '24px Arial';
        this.labelContext.textBaseline = 'middle';
        this.labelContext.fillText(text, 10, LABEL_SIZE.height / 2);

        const texture = new THREE.CanvasTexture(this.labelCanvas);
        texture.generateMipmaps = false;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        
        texturePool.set(text, texture);
        return texture;
    }

    /**
     * Get or create a material for label
     * @param {THREE.Texture} texture - The label texture
     * @returns {THREE.SpriteMaterial} The material
     */
    getMaterial(texture) {
        const key = texture.uuid;
        if (materialPool.has(key)) {
            return materialPool.get(key);
        }

        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthWrite: false,
            sizeAttenuation: true
        });

        materialPool.set(key, material);
        return material;
    }

    createLabel(text, position) {
        try {
            const texture = this.getTexture(text);
            const material = this.getMaterial(texture);
            const sprite = new THREE.Sprite(material);
            
            sprite.position.copy(position);
            sprite.scale.set(LABEL_SCALE.x, LABEL_SCALE.y, LABEL_SCALE.z);
            
            this.scene.add(sprite);
            this.labels.set(text, sprite);
            
            return sprite;
        } catch (error) {
            console.error('Error creating label:', error);
            return null;
        }
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
            
            // Return material and texture to pools
            if (label.material) {
                const texture = label.material.map;
                if (texture) {
                    texturePool.delete(text);
                    texture.dispose();
                }
                materialPool.delete(label.material.uuid);
                label.material.dispose();
            }
            
            this.labels.delete(text);
        }
    }

    updateAll() {
        const cameraPosition = this.camera.position;
        this.labels.forEach(label => {
            label.lookAt(cameraPosition);
        });
    }

    dispose() {
        // Dispose of all labels
        this.labels.forEach((label, text) => {
            this.removeLabel(text);
        });

        // Clear pools
        texturePool.forEach(texture => texture.dispose());
        materialPool.forEach(material => material.dispose());
        
        texturePool.clear();
        materialPool.clear();
        
        // Clear canvas
        this.labelContext.clearRect(0, 0, LABEL_SIZE.width, LABEL_SIZE.height);
        this.labelCanvas.width = 1;
        this.labelCanvas.height = 1;
    }
}

// Detect pinch with strength
function isPinching(hand) {
    try {
        const indexTip = hand.joints['index-finger-tip'];
        const thumbTip = hand.joints['thumb-tip'];

        if (indexTip && thumbTip) {
            const distance = indexTip.position.distanceTo(thumbTip.position);
            const strength = Math.max(0, 1 - (distance / PINCH_THRESHOLD));
            return { isPinched: distance < PINCH_THRESHOLD, strength };
        }
    } catch (error) {
        console.error('Error detecting pinch:', error);
    }
    return { isPinched: false, strength: 0 };
}

// Get or create geometry for pinch indicator
function getPinchIndicatorGeometry() {
    const key = 'pinchIndicator';
    if (geometryPool.has(key)) {
        return geometryPool.get(key);
    }

    const geometry = new THREE.SphereGeometry(0.01, 8, 8);
    geometryPool.set(key, geometry);
    return geometry;
}

// Get or create material for pinch indicator
function getPinchIndicatorMaterial() {
    const key = 'pinchIndicator';
    if (materialPool.has(key)) {
        return materialPool.get(key);
    }

    const material = new THREE.MeshPhongMaterial({
        color: 0x00ff00,
        transparent: true,
        opacity: 0.5,
        depthWrite: false
    });
    materialPool.set(key, material);
    return material;
}

// Create visual feedback sphere for pinch state
function createPinchIndicator() {
    const geometry = getPinchIndicatorGeometry();
    const material = getPinchIndicatorMaterial();
    return new THREE.Mesh(geometry, material);
}

// Update pinch indicator position and appearance
function updatePinchIndicator(hand, indicator) {
    if (!hand?.joints || !indicator) return;

    try {
        const indexTip = hand.joints['index-finger-tip'];
        const thumbTip = hand.joints['thumb-tip'];
        
        if (indexTip && thumbTip) {
            indicator.position.copy(indexTip.position).lerp(thumbTip.position, 0.5);
            const { strength } = isPinching(hand);
            indicator.material.opacity = strength * 0.8;
            indicator.scale.setScalar(1 - (strength * 0.5));
        }
    } catch (error) {
        console.error('Error updating pinch indicator:', error);
    }
}

// Initialize XR interaction
export function initXRInteraction(scene, camera, renderer, onSelect) {
    const xrLabelManager = new XRLabelManager(scene, camera);
    
    // Create default interactable object geometry and material
    const interactableGeometry = new THREE.BoxGeometry(0.1, 0.1, 0.1);
    geometryPool.set('interactable', interactableGeometry);
    
    const interactableMaterial = new THREE.MeshStandardMaterial({
        color: 0xff0000,
        roughness: 0.7,
        metalness: 0.3
    });
    materialPool.set('interactable', interactableMaterial);

    // Create interactable object function
    const createInteractableObject = (position) => {
        const geometry = geometryPool.get('interactable');
        const material = materialPool.get('interactable').clone(); // Clone material for individual control
        
        const object = new THREE.Mesh(geometry, material);
        object.position.copy(position);
        object.userData.interactable = true;
        scene.add(object);
        interactableObjects.add(object);
        return object;
    };

    // Create default objects
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
        try {
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
        } catch (error) {
            console.error('Error setting up hand tracking:', error);
        }
    });

    // Create update function
    const update = () => {
        try {
            // Update both hands
            for (const [handedness, hand] of Object.entries(hands)) {
                if (hand?.joints) {
                    const grabState = grabStates[handedness];
                    const { isPinched, strength } = isPinching(hand);
                    
                    updatePinchIndicator(hand, pinchIndicators[handedness]);

                    if (isPinched && strength > PINCH_STRENGTH_THRESHOLD) {
                        if (!grabState.grabbedObject) {
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
                            const indexTip = hand.joints['index-finger-tip'];
                            grabState.grabbedObject.position.copy(indexTip.position);
                        }
                        grabState.pinching = true;
                    } else if (grabState.pinching) {
                        if (grabState.grabbedObject) {
                            grabState.grabbedObject.userData.isGrabbed = false;
                            grabState.grabbedObject.material.emissive.setHex(0x000000);
                            grabState.grabbedObject = null;
                        }
                        grabState.pinching = false;
                    }
                }
            }

            // Update labels if in XR
            if (renderer.xr.isPresenting) {
                xrLabelManager.updateAll();
            }
        } catch (error) {
            console.error('Error in XR update:', error);
        }
    };

    // Create cleanup function
    const cleanup = () => {
        try {
            // Dispose of all pooled resources
            geometryPool.forEach(geometry => geometry.dispose());
            materialPool.forEach(material => material.dispose());
            
            // Clear pools
            geometryPool.clear();
            materialPool.clear();
            
            // Dispose of pinch indicators
            Object.values(pinchIndicators).forEach(indicator => {
                if (indicator) {
                    if (indicator.geometry) indicator.geometry.dispose();
                    if (indicator.material) indicator.material.dispose();
                    scene.remove(indicator);
                }
            });
            
            // Dispose of hand models
            Object.values(hands).forEach(hand => {
                if (hand) {
                    scene.remove(hand);
                }
            });
            
            // Clear interactable objects
            interactableObjects.clear();
            
            // Dispose of label manager
            xrLabelManager.dispose();
        } catch (error) {
            console.error('Error cleaning up XR resources:', error);
        }
    };

    return {
        hands: Object.values(hands),
        controllers: [],
        xrLabelManager,
        update,
        cleanup,
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
    try {
        // Update both hands
        for (const [handedness, hand] of Object.entries(hands)) {
            if (hand?.joints) {
                const grabState = grabStates[handedness];
                const { isPinched, strength } = isPinching(hand);
                
                updatePinchIndicator(hand, pinchIndicators[handedness]);

                if (isPinched && strength > PINCH_STRENGTH_THRESHOLD) {
                    if (!grabState.grabbedObject) {
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
                        const indexTip = hand.joints['index-finger-tip'];
                        grabState.grabbedObject.position.copy(indexTip.position);
                    }
                    grabState.pinching = true;
                } else if (grabState.pinching) {
                    if (grabState.grabbedObject) {
                        grabState.grabbedObject.userData.isGrabbed = false;
                        grabState.grabbedObject.material.emissive.setHex(0x000000);
                        grabState.grabbedObject = null;
                    }
                    grabState.pinching = false;
                }
            }
        }
    } catch (error) {
        console.error('Error handling XR input:', error);
    }
}
