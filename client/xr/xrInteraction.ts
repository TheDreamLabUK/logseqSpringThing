import * as THREE from 'three';
import { XRHandModelFactory } from 'three/examples/jsm/webxr/XRHandModelFactory.js';
import type { Scene, Camera, WebGLRenderer, Object3D } from 'three';

// Constants for interaction
const PINCH_THRESHOLD = 0.015;
const GRAB_THRESHOLD = 0.08;
const PINCH_STRENGTH_THRESHOLD = 0.7;
const UPDATE_INTERVAL = 200; // 5 FPS (200ms between updates)

interface GrabState {
    grabbedObject: Object3D | null;
    pinching: boolean;
}

interface HandState {
    [key: string]: any;
    joints: {
        [key: string]: THREE.Object3D & {
            position: THREE.Vector3;
        };
    };
}

interface PinchState {
    isPinched: boolean;
    strength: number;
}

interface InteractableMesh extends THREE.Mesh {
    material: THREE.MeshPhongMaterial | THREE.MeshStandardMaterial;
}

interface PositionUpdate {
    nodeId: string;
    position: THREE.Vector3;
    timestamp: number;
}

export class EnhancedXRInteractionHandler {
    private scene: Scene;
    private camera: Camera;
    private renderer: WebGLRenderer;
    private handModelFactory: XRHandModelFactory;
    private hands: { [key: string]: HandState | null };
    private handModels: { [key: string]: Object3D | null };
    private grabStates: { [key: string]: GrabState };
    private pinchIndicators: { [key: string]: THREE.Mesh | null };
    private interactableObjects: Set<Object3D>;
    private materialPool: Map<string, THREE.MeshPhongMaterial>;
    private geometryPool: Map<string, THREE.BufferGeometry>;
    private lastUpdateTime: number;
    private pendingUpdates: Map<string, PositionUpdate>;
    private onPositionUpdate?: (nodeId: string, position: THREE.Vector3) => void;

    constructor(scene: Scene, camera: Camera, renderer: WebGLRenderer) {
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        
        this.handModelFactory = new XRHandModelFactory();
        this.hands = { left: null, right: null };
        this.handModels = { left: null, right: null };
        
        this.grabStates = {
            left: { grabbedObject: null, pinching: false },
            right: { grabbedObject: null, pinching: false }
        };
        
        this.pinchIndicators = { left: null, right: null };
        this.interactableObjects = new Set();
        this.materialPool = new Map();
        this.geometryPool = new Map();
        this.lastUpdateTime = 0;
        this.pendingUpdates = new Map();
        
        this.initResources();
    }

    private initResources(): void {
        const geometry = new THREE.SphereGeometry(0.01, 8, 8);
        this.geometryPool.set('pinchIndicator', geometry);

        const material = new THREE.MeshPhongMaterial({
            color: 0x00ff00,
            transparent: true,
            opacity: 0.5,
            depthWrite: false
        });
        this.materialPool.set('pinchIndicator', material);

        this.pinchIndicators.left = this.createPinchIndicator();
        this.pinchIndicators.right = this.createPinchIndicator();
        this.scene.add(this.pinchIndicators.left!);
        this.scene.add(this.pinchIndicators.right!);
    }

    async initHandTracking(session: XRSession): Promise<void> {
        try {
            for (const handedness of ['left', 'right']) {
                const hand = this.renderer.xr.getHand(handedness === 'left' ? 0 : 1);
                const handModel = this.handModelFactory.createHandModel(hand, 'mesh');
                
                this.hands[handedness] = hand as unknown as HandState;
                this.handModels[handedness] = handModel;
                
                hand.add(handModel);
                this.scene.add(hand);

                hand.addEventListener('pinchstart', () => this.onPinchStart(handedness));
                hand.addEventListener('pinchend', () => this.onPinchEnd(handedness));
            }

            session.addEventListener('handtracking', (event: any) => {
                const hand = event.hand;
                const handedness = hand.handedness;
                
                if (this.handModels[handedness]) {
                    this.handModels[handedness]!.visible = hand.visible;
                }
            });
        } catch (error) {
            console.error('Error initializing hand tracking:', error);
        }
    }

    private createPinchIndicator(): THREE.Mesh {
        const geometry = this.geometryPool.get('pinchIndicator')!;
        const material = this.materialPool.get('pinchIndicator')!.clone();
        return new THREE.Mesh(geometry, material);
    }

    private updatePinchIndicator(hand: HandState, indicator: THREE.Mesh): void {
        if (!hand?.joints || !indicator) return;

        try {
            const indexTip = hand.joints['index-finger-tip'];
            const thumbTip = hand.joints['thumb-tip'];
            
            if (indexTip && thumbTip && indicator.material instanceof THREE.MeshPhongMaterial) {
                indicator.position.copy(indexTip.position).lerp(thumbTip.position, 0.5);
                const { strength } = this.isPinching(hand);
                indicator.material.opacity = strength * 0.8;
                indicator.scale.setScalar(1 - (strength * 0.5));
            }
        } catch (error) {
            console.error('Error updating pinch indicator:', error);
        }
    }

    private isPinching(hand: HandState): PinchState {
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

    private onPinchStart(handedness: string): void {
        const hand = this.hands[handedness];
        const grabState = this.grabStates[handedness];

        if (!hand || grabState.pinching) return;

        try {
            const indexTip = hand.joints['index-finger-tip'];
            
            let closestObject: Object3D | null = null;
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

            if (closestObject && this.isInteractableMesh(closestObject)) {
                grabState.grabbedObject = closestObject;
                closestObject.userData.isGrabbed = true;
                
                if (closestObject.material instanceof THREE.MeshStandardMaterial) {
                    closestObject.material.emissive.setHex(0x222222);
                }
            }

            grabState.pinching = true;
        } catch (error) {
            console.error('Error handling pinch start:', error);
        }
    }

    private onPinchEnd(handedness: string): void {
        const grabState = this.grabStates[handedness];

        if (!grabState.pinching) return;

        try {
            if (grabState.grabbedObject && this.isInteractableMesh(grabState.grabbedObject)) {
                grabState.grabbedObject.userData.isGrabbed = false;
                if (grabState.grabbedObject.material instanceof THREE.MeshStandardMaterial) {
                    grabState.grabbedObject.material.emissive.setHex(0x000000);
                }
                grabState.grabbedObject = null;
            }

            grabState.pinching = false;
        } catch (error) {
            console.error('Error handling pinch end:', error);
        }
    }

    private isInteractableMesh(object: Object3D): object is InteractableMesh {
        return object instanceof THREE.Mesh && 
               (object.material instanceof THREE.MeshPhongMaterial || 
                object.material instanceof THREE.MeshStandardMaterial);
    }

    setPositionUpdateCallback(callback: (nodeId: string, position: THREE.Vector3) => void): void {
        this.onPositionUpdate = callback;
    }

    private processPendingUpdates(): void {
        const now = performance.now();
        if (now - this.lastUpdateTime < UPDATE_INTERVAL) {
            return;
        }

        this.pendingUpdates.forEach((update, nodeId) => {
            if (this.onPositionUpdate) {
                this.onPositionUpdate(nodeId, update.position);
            }
        });

        this.pendingUpdates.clear();
        this.lastUpdateTime = now;
    }

    makeInteractable(object: Object3D): void {
        object.userData.interactable = true;
        this.interactableObjects.add(object);
    }

    removeInteractable(object: Object3D): void {
        object.userData.interactable = false;
        this.interactableObjects.delete(object);
    }

    update(): void {
        try {
            for (const [handedness, hand] of Object.entries(this.hands)) {
                if (hand?.joints) {
                    const grabState = this.grabStates[handedness];
                    const { isPinched, strength } = this.isPinching(hand);
                    
                    this.updatePinchIndicator(hand, this.pinchIndicators[handedness]!);

                    if (isPinched && strength > PINCH_STRENGTH_THRESHOLD) {
                        if (grabState.grabbedObject) {
                            const indexTip = hand.joints['index-finger-tip'];
                            grabState.grabbedObject.position.copy(indexTip.position);

                            // Queue position update
                            if (grabState.grabbedObject.userData.id) {
                                this.pendingUpdates.set(grabState.grabbedObject.userData.id, {
                                    nodeId: grabState.grabbedObject.userData.id,
                                    position: grabState.grabbedObject.position.clone(),
                                    timestamp: performance.now()
                                });
                            }
                        }
                    }
                }
            }

            // Process pending updates at 5 FPS
            this.processPendingUpdates();
        } catch (error) {
            console.error('Error in XR interaction update:', error);
        }
    }

    cleanup(): void {
        try {
            this.geometryPool.forEach(geometry => geometry.dispose());
            this.geometryPool.clear();

            this.materialPool.forEach(material => material.dispose());
            this.materialPool.clear();

            Object.values(this.pinchIndicators).forEach(indicator => {
                if (indicator) {
                    if (indicator.geometry) indicator.geometry.dispose();
                    if (indicator.material instanceof THREE.Material) {
                        indicator.material.dispose();
                    }
                    this.scene.remove(indicator);
                }
            });

            Object.values(this.hands).forEach(hand => {
                if (hand) {
                    this.scene.remove(hand as unknown as Object3D);
                }
            });

            this.interactableObjects.clear();
            this.grabStates.left = { grabbedObject: null, pinching: false };
            this.grabStates.right = { grabbedObject: null, pinching: false };
            this.pendingUpdates.clear();
        } catch (error) {
            console.error('Error cleaning up XR interaction:', error);
        }
    }
}

export function initXRInteraction(
    scene: Scene, 
    camera: Camera, 
    renderer: WebGLRenderer, 
    onPositionUpdate?: (nodeId: string, position: THREE.Vector3) => void
): EnhancedXRInteractionHandler {
    const handler = new EnhancedXRInteractionHandler(scene, camera, renderer);
    if (onPositionUpdate) {
        handler.setPositionUpdateCallback(onPositionUpdate);
    }
    return handler;
}
