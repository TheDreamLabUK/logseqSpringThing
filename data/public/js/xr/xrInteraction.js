import * as THREE from 'three';
import { XRHandModelFactory } from 'three/examples/jsm/webxr/XRHandModelFactory.js';

// Constants for interaction
const PINCH_THRESHOLD = 0.015;
const GRAB_THRESHOLD = 0.08;
const PINCH_STRENGTH_THRESHOLD = 0.7;
const LABEL_SIZE = { width: 256, height: 128 };
const LABEL_SCALE = { x: 0.5, y: 0.25, z: 1 };

/**
 * Enhanced XR Interaction Handler
 */
class EnhancedXRInteractionHandler {
    constructor(scene, camera, renderer) {
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        
        // Hand tracking
        this.handModelFactory = new XRHandModelFactory();
        this.hands = { left: null, right: null };
        this.handModels = { left: null, right: null };
        
        // Interaction states
        this.grabStates = {
            left: { grabbedObject: null, pinching: false },
            right: { grabbedObject: null, pinching: false }
        };
        
        // Visual feedback
        this.pinchIndicators = { left: null, right: null };
        
        // Interactable objects
        this.interactableObjects = new Set();
        
        // Resource pools
        this.materialPool = new Map();
        this.geometryPool = new Map();
        
        // Initialize resources
        this.initResources();
    }

    /**
     * Initialize shared resources
     */
    initResources() {
        // Create pinch indicator geometry
        const geometry = new THREE.SphereGeometry(0.01, 8, 8);
        this.geometryPool.set('pinchIndicator', geometry);

        // Create pinch indicator material
        const material = new THREE.MeshPhongMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.5,
            depthWrite: false
        });
        this.materialPool.set('pinchIndicator', material);

        // Create pinch indicators
        this.pinchIndicators.left = this.createPinchIndicator();
        this.pinchIndicators.right = this.createPinchIndicator();
        this.scene.add(this.pinchIndicators.left);
        this.scene.add(this.pinchIndicators.right);
    }

    /**
     * Initialize hand tracking
     * @param {XRSession} session - The XR session
     */
    async initHandTracking(session) {
        try {
            // Set up hand tracking
            for (const handedness of ['left', 'right']) {
                const hand = this.renderer.xr.getHand(handedness === 'left' ? 0 : 1);
                const handModel = this.handModelFactory.createHandModel(hand, 'mesh');
                
                this.hands[handedness] = hand;
                this.handModels[handedness] = handModel;
                
                hand.add(handModel);
                this.scene.add(hand);

                // Add hand input event listeners
                hand.addEventListener('pinchstart', () => this.onPinchStart(handedness));
                hand.addEventListener('pinchend', () => this.onPinchEnd(handedness));
            }

            // Set up hand tracking events
            session.addEventListener('handtracking', (event) => {
                const hand = event.hand;
                const handedness = hand.handedness;
                
                // Update hand model visibility
                if (this.handModels[handedness]) {
                    this.handModels[handedness].visible = hand.visible;
                }
            });
        } catch (error) {
            console.error('Error initializing hand tracking:', error);
        }
    }

    /**
     * Create visual feedback for pinch state
     * @returns {THREE.Mesh} Pinch indicator mesh
     */
    createPinchIndicator() {
        const geometry = this.geometryPool.get('pinchIndicator');
        const material = this.materialPool.get('pinchIndicator').clone();
        return new THREE.Mesh(geometry, material);
    }

    /**
     * Update pinch indicator position and appearance
     * @param {XRHand} hand - The XR hand
     * @param {THREE.Mesh} indicator - The pinch indicator mesh
     */
    updatePinchIndicator(hand, indicator) {
        if (!hand?.joints || !indicator) return;

        try {
            const indexTip = hand.joints['index-finger-tip'];
            const thumbTip = hand.joints['thumb-tip'];
            
            if (indexTip && thumbTip) {
                // Position indicator between finger and thumb
                indicator.position.copy(indexTip.position).lerp(thumbTip.position, 0.5);
                
                // Update appearance based on pinch strength
                const { strength } = this.isPinching(hand);
                indicator.material.opacity = strength * 0.8;
                indicator.scale.setScalar(1 - (strength * 0.5));
            }
        } catch (error) {
            console.error('Error updating pinch indicator:', error);
        }
    }

    /**
     * Check if hand is performing pinch gesture
     * @param {XRHand} hand - The XR hand
     * @returns {object} Pinch state and strength
     */
    isPinching(hand) {
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

    /**
     * Handle pinch start event
     * @param {string} handedness - The hand that started pinching
     */
    onPinchStart(handedness) {
        const hand = this.hands[handedness];
        const grabState = this.grabStates[handedness];

        if (!hand || grabState.pinching) return;

        try {
            const indexTip = hand.joints['index-finger-tip'];
            
            // Find closest interactable object
            let closestObject = null;
            let closestDistance = GRAB_THRESHOLD;

            for (const object of this.interactableObjects) {
                if (!object.userData.isGrabbed) {
                    const distance = indexTip.position.distanceTo(object.position);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestObject = object;
                    }
                }
            }

            if (closestObject) {
                grabState.grabbedObject = closestObject;
                closestObject.userData.isGrabbed = true;
                
                // Highlight grabbed object
                if (closestObject.material?.emissive) {
                    closestObject.material.emissive.setHex(0x222222);
                }
            }

            grabState.pinching = true;
        } catch (error) {
            console.error('Error handling pinch start:', error);
        }
    }

    /**
     * Handle pinch end event
     * @param {string} handedness - The hand that ended pinching
     */
    onPinchEnd(handedness) {
        const grabState = this.grabStates[handedness];

        if (!grabState.pinching) return;

        try {
            if (grabState.grabbedObject) {
                grabState.grabbedObject.userData.isGrabbed = false;
                if (grabState.grabbedObject.material?.emissive) {
                    grabState.grabbedObject.material.emissive.setHex(0x000000);
                }
                grabState.grabbedObject = null;
            }

            grabState.pinching = false;
        } catch (error) {
            console.error('Error handling pinch end:', error);
        }
    }

    /**
     * Make an object interactable
     * @param {THREE.Object3D} object - The object to make interactable
     */
    makeInteractable(object) {
        object.userData.interactable = true;
        this.interactableObjects.add(object);
    }

    /**
     * Remove interactable status from object
     * @param {THREE.Object3D} object - The object to remove
     */
    removeInteractable(object) {
        object.userData.interactable = false;
        this.interactableObjects.delete(object);
    }

    /**
     * Update interaction state
     */
    update() {
        try {
            // Update both hands
            for (const [handedness, hand] of Object.entries(this.hands)) {
                if (hand?.joints) {
                    const grabState = this.grabStates[handedness];
                    const { isPinched, strength } = this.isPinching(hand);
                    
                    this.updatePinchIndicator(hand, this.pinchIndicators[handedness]);

                    if (isPinched && strength > PINCH_STRENGTH_THRESHOLD) {
                        if (grabState.grabbedObject) {
                            // Update grabbed object position
                            const indexTip = hand.joints['index-finger-tip'];
                            grabState.grabbedObject.position.copy(indexTip.position);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error in XR interaction update:', error);
        }
    }

    /**
     * Clean up resources
     */
    cleanup() {
        try {
            // Dispose of geometries
            this.geometryPool.forEach(geometry => geometry.dispose());
            this.geometryPool.clear();

            // Dispose of materials
            this.materialPool.forEach(material => material.dispose());
            this.materialPool.clear();

            // Remove pinch indicators
            Object.values(this.pinchIndicators).forEach(indicator => {
                if (indicator) {
                    if (indicator.geometry) indicator.geometry.dispose();
                    if (indicator.material) indicator.material.dispose();
                    this.scene.remove(indicator);
                }
            });

            // Remove hand models
            Object.values(this.hands).forEach(hand => {
                if (hand) {
                    this.scene.remove(hand);
                }
            });

            // Clear collections
            this.interactableObjects.clear();
            this.grabStates.left = { grabbedObject: null, pinching: false };
            this.grabStates.right = { grabbedObject: null, pinching: false };
        } catch (error) {
            console.error('Error cleaning up XR interaction:', error);
        }
    }
}

// Export functions
export function initXRInteraction(scene, camera, renderer) {
    return new EnhancedXRInteractionHandler(scene, camera, renderer);
}

export function handleXRInput(frame, referenceSpace) {
    // This function is now handled internally by EnhancedXRInteractionHandler
    // Left for backward compatibility
}
