/**
 * XR interaction handling for controllers and hands
 */

import * as THREE from 'three';
import { XRSessionManager } from './xrSessionManager';
import { NodeManager } from '../rendering/nodes';
import { settingsManager } from '../state/settings';
import { createLogger } from '../core/utils';

// Logger will be used for debugging XR interactions and haptic feedback
const _logger = createLogger('XRInteraction');

// XR Interaction Action Types
type XRInteractionActionType = 'select' | 'squeeze';

// Extended XR types
interface XRHandWithHaptics extends XRHand {
    vibrate?(intensity: number, duration: number): void;
}

interface XRSessionWithPose extends XRSession {
    getPose?(source: XRSpace, referenceSpace: XRReferenceSpace): XRPose | undefined;
}

interface XRInteractionActionEvent extends THREE.Event {
    type: `${XRInteractionActionType}start` | `${XRInteractionActionType}end`;
    data: XRInputSource;
    hand?: XRHandWithHaptics;
}

interface XRController extends THREE.Group {
    userData: {
        inputSource: XRInputSource;
    };
}

interface XRNodeMeshUserData {
    nodeId: string;
}

interface XRNodeMesh extends THREE.Object3D {
    userData: XRNodeMeshUserData;
}

declare module 'three' {
    interface Object3DEventMap {
        selectstart: XRInteractionActionEvent;
        selectend: XRInteractionActionEvent;
        squeezestart: XRInteractionActionEvent;
        squeezeend: XRInteractionActionEvent;
    }
}

type XRInteractor = XRController | XRHandWithHaptics;

// Type guard function to check if an object is an XRNodeMesh
function isXRNodeMesh(obj: THREE.Object3D): obj is XRNodeMesh {
    return (
        obj !== null &&
        typeof obj === 'object' &&
        'userData' in obj &&
        obj.userData !== null &&
        typeof obj.userData === 'object' &&
        'nodeId' in obj.userData &&
        typeof obj.userData.nodeId === 'string'
    );
}

export class XRInteraction {
    private static instance: XRInteraction;
    private xrManager: XRSessionManager;
    private nodeManager: NodeManager;

    // Interaction state
    private selectedNode: string | null = null;
    private hoveredNode: string | null = null;
    private isGrabbing: boolean = false;
    private lastInteractorPosition: THREE.Vector3;
    private grabOffset: THREE.Vector3;

    // Raycasting
    private raycaster: THREE.Raycaster;
    private tempMatrix: THREE.Matrix4;

    // Hand Tracking
    private hands: XRHandWithHaptics[] = [];
    private pinchThreshold: number = 0.025;

    private constructor(xrManager: XRSessionManager, nodeManager: NodeManager) {
        this.xrManager = xrManager;
        this.nodeManager = nodeManager;

        this.lastInteractorPosition = new THREE.Vector3();
        this.grabOffset = new THREE.Vector3();
        this.raycaster = new THREE.Raycaster();
        this.tempMatrix = new THREE.Matrix4();
        this.setupEventListeners();
    }

    static getInstance(xrManager: XRSessionManager, nodeManager: NodeManager): XRInteraction {
        if (!XRInteraction.instance) {
            XRInteraction.instance = new XRInteraction(xrManager, nodeManager);
        }
        return XRInteraction.instance;
    }

    private setupEventListeners(): void {
        const session = this.xrManager.getSession();
        if (!session) return;

        // Handle controller / hand updates
        session.addEventListener('inputsourceschange', (event: XRInputSourcesChangeEvent) => {
            // Clear old event listeners
            this.xrManager.getControllers().forEach(controller => {
                controller.removeEventListener('selectstart', this.handleSelectStart);
                controller.removeEventListener('selectend', this.handleSelectEnd);
                controller.removeEventListener('squeezestart', this.handleSqueezeStart);
                controller.removeEventListener('squeezeend', this.handleSqueezeEnd);
            });

            this.hands = [];
            event.added.forEach((source: XRInputSource) => {
                if (source.hand) {
                    this.hands.push(source.hand as XRHandWithHaptics);
                }
            });

            // Setup new input sources
            this.xrManager.getControllers().forEach((controller) => {
                controller.addEventListener('selectstart', this.handleSelectStart);
                controller.addEventListener('selectend', this.handleSelectEnd);
                controller.addEventListener('squeezestart', this.handleSqueezeStart);
                controller.addEventListener('squeezeend', this.handleSqueezeEnd);
            });
        });

        // Initial Setup
        this.xrManager.getControllers().forEach((controller) => {
            controller.addEventListener('selectstart', this.handleSelectStart);
            controller.addEventListener('selectend', this.handleSelectEnd);
            controller.addEventListener('squeezestart', this.handleSqueezeStart);
            controller.addEventListener('squeezeend', this.handleSqueezeEnd);
        });
    }

    private handleSelectStart = (event: THREE.Event) => {
        if (this.hoveredNode) {
            const interactionEvent = event as XRInteractionActionEvent;
            this.startGrab(this.hoveredNode, interactionEvent.data, interactionEvent.hand);
        }
    }

    private handleSelectEnd = () => {
        this.endGrab();
    }

    private handleSqueezeStart = (event: THREE.Event) => {
        if (this.hoveredNode) {
            const interactionEvent = event as XRInteractionActionEvent;
            this.startGrab(this.hoveredNode, interactionEvent.data, interactionEvent.hand);
        }
    }

    private handleSqueezeEnd = () => {
        this.endGrab();
    }

    /**
     * Update interaction state
     */
    update(frame: XRFrame): void {
        const session = this.xrManager.getSession() as XRSessionWithPose;
        const referenceSpace = this.xrManager.getReferenceSpace();

        if (!session || !referenceSpace) return;

        // Update controller interaction
        this.xrManager.getControllers().forEach((baseController) => {
            const controller = baseController as XRController;
            const inputSource = controller.userData.inputSource;
            if (!inputSource) return;

            // Get controller pose
            const pose = frame.getPose(inputSource.targetRaySpace, referenceSpace);
            if (!pose) return;

            // Update raycaster
            controller.updateMatrixWorld();
            this.tempMatrix.identity().extractRotation(controller.matrixWorld);
            this.raycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
            this.raycaster.ray.direction.set(0, 0, -1).applyMatrix4(this.tempMatrix);

            // Check for intersections with nodes
            this.checkNodeIntersections(controller);

            // Update grabbed node position
            if (this.isGrabbing && this.selectedNode) {
                this.updateGrabbedNodePosition(controller);
            }

            // Provide haptic feedback if enabled
            if (inputSource.gamepad && settingsManager.getSettings().xrControllerVibration) {
                this.handleHapticFeedback(inputSource.gamepad);
            }
        });

        // Handle Hand Interactions
        this.hands.forEach((hand) => {
            this.checkHandIntersections(hand, frame, referenceSpace);

            if (this.isGrabbing && this.selectedNode) {
                this.updateGrabbedNodePosition(hand);
            }
        });
    }

    private checkHandIntersections(hand: XRHandWithHaptics, frame: XRFrame, referenceSpace: XRReferenceSpace): void {
        if (!hand || !frame || !referenceSpace) return;

        // Get index and thumb tip
        const indexTipPose = frame.getPose(hand.get("index-finger-tip") as XRSpace, referenceSpace);
        const thumbTipPose = frame.getPose(hand.get("thumb-tip") as XRSpace, referenceSpace);
        if (!indexTipPose || !thumbTipPose) return;

        const indexTipPosition = new THREE.Vector3().fromArray(indexTipPose.transform.matrix.slice(12, 15));
        const thumbTipPosition = new THREE.Vector3().fromArray(thumbTipPose.transform.matrix.slice(12, 15));
        const distance = indexTipPosition.distanceTo(thumbTipPosition);

        let closestNode: XRNodeMesh | null = null;
        let minDistance = Infinity;

        // Get meshes and check each one
        const meshes = this.nodeManager.getAllNodeMeshes();
        for (const mesh of meshes) {
            if (isXRNodeMesh(mesh)) {
                const nodePosition = new THREE.Vector3().setFromMatrixPosition(mesh.matrixWorld);
                const nodeDistance = nodePosition.distanceTo(indexTipPosition);

                if (nodeDistance < minDistance) {
                    minDistance = nodeDistance;
                    closestNode = mesh;
                }
            }
        }

        if (minDistance < 0.1 && closestNode) {
            const nodeId = closestNode.userData.nodeId;
            if (nodeId !== this.hoveredNode) {
                // Update hover state
                this.hoveredNode = nodeId;
                this.nodeManager.highlightNode(nodeId);
                // Trigger haptic pulse for hover
                this.pulseHand(hand, 0.2, 50);
            }
            if (distance < this.pinchThreshold && !this.isGrabbing) {
                this.startGrab(nodeId, hand, hand);
            }
        } else if (this.hoveredNode) {
            // Clear hover state
            this.nodeManager.highlightNode(null);
            this.hoveredNode = null;
        }
    }

    private checkNodeIntersections(controller: XRController): void {
        // Get closest intersection
        const meshes = this.nodeManager.getAllNodeMeshes();
        const intersects = this.raycaster.intersectObjects(meshes);

        if (intersects.length > 0) {
            const intersectedObject = intersects[0].object;
            if (isXRNodeMesh(intersectedObject)) {
                const nodeId = intersectedObject.userData.nodeId;
                if (nodeId !== this.hoveredNode) {
                    // Update hover state
                    this.hoveredNode = nodeId;
                    this.nodeManager.highlightNode(nodeId);

                    // Trigger haptic pulse for hover
                    this.pulseController(controller, 0.2, 50);
                }
            }
        } else if (this.hoveredNode) {
            // Clear hover state
            this.nodeManager.highlightNode(null);
            this.hoveredNode = null;
        }
    }

    private startGrab(nodeId: string, interactor: XRInputSource | XRHandWithHaptics, hand?: XRHandWithHaptics): void {
        this.selectedNode = nodeId;
        this.isGrabbing = true;

        // Store initial grab position
        if (interactor instanceof THREE.Group) {
            const controller = interactor as unknown as XRController;
            this.lastInteractorPosition.setFromMatrixPosition(controller.matrixWorld);
            this.pulseController(controller, 0.7, 100);
        } else if (hand) {
            const wrist = hand.get("wrist") as XRSpace;
            const session = this.xrManager.getSession() as XRSessionWithPose;
            const pose = session?.getPose?.(wrist, this.xrManager.getReferenceSpace() as XRReferenceSpace);
            if (pose) {
                this.lastInteractorPosition.fromArray(pose.transform.matrix.slice(12, 15));
            }
            this.pulseHand(hand, 0.7, 100);
        }

        // Calculate grab offset
        const nodePosition = this.nodeManager.getNodePosition(nodeId);
        this.grabOffset.subVectors(nodePosition, this.lastInteractorPosition);
    }

    private endGrab(): void {
        if (this.isGrabbing) {
            this.isGrabbing = false;
            this.selectedNode = null;
        }
    }

    private updateGrabbedNodePosition(interactor: XRInteractor): void {
        if (!this.selectedNode) return;

        // Get current interactor position
        const currentPosition = new THREE.Vector3();
        if (interactor instanceof THREE.Group) {
            currentPosition.setFromMatrixPosition(interactor.matrixWorld);
        } else {
            const wrist = interactor.get("wrist") as XRSpace;
            const session = this.xrManager.getSession() as XRSessionWithPose;
            const pose = session?.getPose?.(wrist, this.xrManager.getReferenceSpace() as XRReferenceSpace);
            if (pose) {
                currentPosition.fromArray(pose.transform.matrix.slice(12, 15));
            }
        }

        // Calculate new node position
        const newPosition = currentPosition.clone().add(this.grabOffset);
        this.nodeManager.updateNodePosition(this.selectedNode, newPosition);

        // Update last position
        this.lastInteractorPosition.copy(currentPosition);
    }

    private handleHapticFeedback(__gamepad: Gamepad): void {
        if (!settingsManager.getSettings().xrControllerVibration) return;

        // Add haptic feedback logic based on interactions
        // For example, vibrate when near nodes or when grabbing
    }

    private pulseController(controller: XRController, intensity: number, duration: number): void {
        const inputSource = controller.userData.inputSource;
        if (!inputSource?.gamepad || !settingsManager.getSettings().xrControllerVibration) return;

        const actuator = inputSource.gamepad.hapticActuators?.[0] as any;
        if (actuator) {
            try {
                actuator.pulse(intensity, duration);
            } catch (error) {
                _logger.warn('Haptic feedback not supported:', error);
            }
        }
    }

    private pulseHand(hand: XRHandWithHaptics, intensity: number, duration: number): void {
        if (!settingsManager.getSettings().xrControllerVibration) return;
        try {
            hand.vibrate?.(intensity, duration);
        } catch (error) {
            _logger.warn('Haptic feedback not supported for hands:', error);
        }
    }

    /**
     * Clean up resources
     */
    dispose(): void {
        this.endGrab();
        this.hoveredNode = null;
        this.selectedNode = null;
    }
}
